#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx2[http2,brotli,zstd]",
#     "anyio",
#     "rich",
#     "typer",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run download_images.py [ARGS]
# 3. Or make executable and run:
#      chmod +x download_images.py && ./download_images.py
#
# Flags:
#   --csv PATH      input CSV (default: datasetStrigidae_filtered.csv)
#   --out DIR       output dir (default: images/)
#   --workers INT   concurrent downloads (default: 16)
#   --limit INT     process the first N data rows in CSV order; 0 = all (default: 0)
#   --overwrite     re-download files that already exist (default: skip them)
#
# Layout: <out>/<sanitized scientific_name>/<id><ext> plus <out>/manifest.csv.
# Resume-safe: existing files with size > 0 are skipped unless --overwrite.
# ──────────────────

from __future__ import annotations

import csv, logging, re, socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, NewType, assert_never, override
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
import httpx2
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeRemainingColumn

ImageId = NewType("ImageId", int)
TaxonId = NewType("TaxonId", str)

DEFAULT_CSV: Final = Path("/home/kauan/Projetos/IAcoruja/datasetStrigidae_filtered.csv")
DEFAULT_OUT: Final = Path("/home/kauan/Projetos/IAcoruja/images")
REQUIRED_COLUMNS: Final = ("id", "image_url", "scientific_name", "common_name", "taxon_id")
MANIFEST_FIELDS: Final = ("id", "scientific_name", "common_name", "taxon_id", "image_url", "relative_path", "status", "error")
_UNSAFE_CHARS: Final = frozenset('/\\:*?"<>|')

_LIMITS: Final = httpx2.Limits(max_connections=200, max_keepalive_connections=40, keepalive_expiry=30.0)
_TIMEOUT: Final = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
_SOCKET_OPTIONS: Final = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]

console = Console()
logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=False)


class Status(StrEnum):
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MissingColumnsError(Exception):
    missing: tuple[str, ...]

    @override
    def __str__(self) -> str:
        return f"CSV is missing required columns: {', '.join(self.missing)}"


@dataclass(frozen=True, slots=True)
class RowParseError(Exception):
    line: int
    reason: str

    @override
    def __str__(self) -> str:
        return f"row {self.line}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ImageRow:
    image_id: ImageId
    image_url: str
    scientific_name: str
    common_name: str
    taxon_id: TaxonId
    occurrence: int = 1


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    image_id: ImageId
    scientific_name: str
    common_name: str
    taxon_id: TaxonId
    image_url: str
    relative_path: str
    status: Status
    error: str = ""


@dataclass(frozen=True, slots=True)
class Job:
    """Shared per-run state; results list mutates by design (one writer per index)."""

    results: list[ManifestRecord | None]
    task_id: TaskID
    limiter: anyio.CapacityLimiter
    client: httpx2.AsyncClient
    out: Path
    overwrite: bool
    progress: Progress


def sanitize_folder(name: str, taxon_id: TaxonId) -> str:
    """Deterministic safe folder name; falls back to taxon_id, then 'unknown'."""
    cleaned = _clean(name)
    return cleaned or _clean(str(taxon_id)) or "unknown"


def _clean(text: str) -> str:
    collapsed = re.sub(r"\s+", "_", text.strip())
    replaced = "".join("_" if ch in _UNSAFE_CHARS or ord(ch) < 32 or ord(ch) == 127 else ch for ch in collapsed)
    return re.sub(r"_+", "_", replaced).strip("_.")


def derive_ext(url: str) -> str:
    """Lowercased extension from the URL path; '.jpg' when none is usable."""
    suffix = Path(urlsplit(url).path).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,4}", suffix) else ".jpg"


def write_atomic(target: Path, content: bytes) -> None:
    """Write via temp file + rename so failures never leave a partial image."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid4().hex[:8]}.tmp")
    try:
        _ = tmp.write_bytes(content)
        _ = tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def create_async_client() -> httpx2.AsyncClient:
    """Async client with production defaults (HTTP/2, retries, tuned pool, TCP_NODELAY)."""
    transport = httpx2.AsyncHTTPTransport(
        http2=True,
        retries=3,
        limits=_LIMITS,
        socket_options=_SOCKET_OPTIONS,
    )
    return httpx2.AsyncClient(
        transport=transport,
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"user-agent": "download-images/1.0 (iNaturalist image batch downloader)"},
    )


def load_rows(path: Path, limit: int) -> list[ImageRow]:
    """Parse the CSV once, validating required columns; first `limit` rows in CSV order (0 = all)."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = tuple(col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or []))
        if missing:
            raise MissingColumnsError(missing)
        occurrences: dict[ImageId, int] = {}
        rows: list[ImageRow] = []
        for line_no, raw in enumerate(reader, start=2):
            try:
                image_id = ImageId(int(raw["id"]))
            except ValueError as exc:
                raise RowParseError(line_no, f"invalid id {raw['id']!r}") from exc
            occurrence = occurrences.get(image_id, 0) + 1
            occurrences[image_id] = occurrence
            rows.append(ImageRow(image_id, raw["image_url"], raw["scientific_name"], raw["common_name"], TaxonId(raw["taxon_id"]), occurrence))
    return rows if limit <= 0 else rows[:limit]


def _record(row: ImageRow, relative: str, status: Status, error: str = "") -> ManifestRecord:
    return ManifestRecord(row.image_id, row.scientific_name, row.common_name, row.taxon_id, row.image_url, relative, status, error)


def _error_message(exc: httpx2.HTTPStatusError | httpx2.TimeoutException | httpx2.RequestError) -> str:
    """Compact, stable message for a caught transport error."""
    match exc:
        case httpx2.HTTPStatusError() as status_error:
            return f"HTTP {status_error.response.status_code}"
        case httpx2.TimeoutException() as timeout_error:
            return f"timeout: {timeout_error}"[:200]
        case httpx2.RequestError() as request_error:
            return f"request error: {request_error}"[:200]
        case _:
            assert_never(exc)


def status_style(status: Status) -> str:
    """Rich markup for a download status."""
    match status:
        case Status.DOWNLOADED: return "[green]downloaded[/green]"
        case Status.SKIPPED: return "[yellow]skipped[/yellow]"
        case Status.ERROR: return "[red]error[/red]"
        case _: assert_never(status)


def write_manifest_sync(out: Path, entries: list[ManifestRecord]) -> None:
    """Write manifest.csv with the exact required column order."""
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({"id": entry.image_id, "scientific_name": entry.scientific_name, "common_name": entry.common_name, "taxon_id": entry.taxon_id, "image_url": entry.image_url, "relative_path": entry.relative_path, "status": entry.status.value, "error": entry.error})


async def write_manifest_async(out: Path, results: list[ManifestRecord | None], rows: list[ImageRow]) -> list[ManifestRecord]:
    """Mark unfinished slots as aborted errors, sort by numeric id, write manifest off-loop."""
    entries = [result if result is not None else _record(row, "", Status.ERROR, "aborted") for result, row in zip(results, rows, strict=True)]
    entries.sort(key=lambda entry: entry.image_id)
    await anyio.to_thread.run_sync(write_manifest_sync, out, entries)
    return entries


async def _worker(idx: int, row: ImageRow, job: Job) -> None:
    """Download one image and record its manifest entry (per-item failures never abort the run)."""
    async with job.limiter:
        job.results[idx] = await _download_one(row, job)
        job.progress.advance(job.task_id)


async def _download_one(row: ImageRow, job: Job) -> ManifestRecord:
    """Fetch one row's image to <out>/<species>/<id><ext>; skip or record an error as appropriate."""
    folder = sanitize_folder(row.scientific_name, row.taxon_id)
    ext = derive_ext(row.image_url)
    stem = f"{row.image_id}" if row.occurrence == 1 else f"{row.image_id}~{row.occurrence}"
    relative = f"{folder}/{stem}{ext}"
    target = job.out / folder / f"{stem}{ext}"
    if not job.overwrite and target.is_file() and target.stat().st_size > 0:
        return _record(row, relative, Status.SKIPPED)
    try:
        response = await job.client.get(row.image_url)
        _ = response.raise_for_status()
        content = await response.aread()
        await anyio.to_thread.run_sync(write_atomic, target, content)
    except (httpx2.HTTPStatusError, httpx2.TimeoutException, httpx2.RequestError) as exc:
        return _record(row, relative, Status.ERROR, _error_message(exc))
    except OSError as exc:
        return _record(row, relative, Status.ERROR, f"write failed: {exc}")
    return _record(row, relative, Status.DOWNLOADED)


def print_summary(entries: list[ManifestRecord]) -> None:
    """Print per-status counts."""
    counts: dict[Status, int] = {status: 0 for status in Status}
    for entry in entries:
        counts[entry.status] += 1
    console.print("[bold]Summary:[/bold] " + "  ".join(f"{status_style(s)}: [bold]{counts[s]}[/bold]" for s in Status))


def _run(csv_path: Path, out: Path, workers: int, limit: int, overwrite: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    rows = load_rows(csv_path, limit)
    out.mkdir(parents=True, exist_ok=True)
    console.print(f"Processing [bold]{len(rows)}[/bold] rows from [cyan]{csv_path}[/cyan] → [cyan]{out}[/cyan]")
    progress = Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        MofNCompleteColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Downloading images", total=len(rows))
        job = Job(
            results=[None] * len(rows), task_id=task, limiter=anyio.CapacityLimiter(workers),
            client=create_async_client(), out=out, overwrite=overwrite, progress=progress,
        )
        entries = anyio.run(_run_async, rows, job)
    console.print(f"[bold]Done.[/bold] manifest written to [cyan]{out / 'manifest.csv'}[/cyan]")
    print_summary(entries)


async def _run_async(rows: list[ImageRow], job: Job) -> list[ManifestRecord]:
    entries: list[ManifestRecord] = []
    try:
        async with job.client:
            async with anyio.create_task_group() as tg:
                for idx, row in enumerate(rows):
                    _ = tg.start_soon(_worker, idx, row, job)
    finally:
        with anyio.CancelScope(shield=True):
            entries = await write_manifest_async(job.out, job.results, rows)
    return entries


@app.command()
def main(
    csv_path: Annotated[Path, typer.Option("--csv", help="Input CSV with image rows.")] = DEFAULT_CSV,
    out: Annotated[Path, typer.Option("--out", help="Output dir: <species>/<id><ext> plus manifest.csv.")] = DEFAULT_OUT,
    workers: Annotated[int, typer.Option("--workers", min=1, max=128, help="Concurrent downloads.")] = 16,
    limit: Annotated[int, typer.Option("--limit", min=0, help="Process the first N data rows in CSV order; 0 = all.")] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Re-download files that already exist (default: skip).")] = False,
) -> None:
    """Download every row's image_url into a folder-per-species layout with a manifest."""
    try:
        _run(csv_path, out, workers, limit, overwrite)
    except (MissingColumnsError, RowParseError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception:  # noqa: BROAD_EXCEPT_OK
        logger.exception("unexpected failure (manifest was still written if a run had started)")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
