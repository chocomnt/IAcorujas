import os
import cv2
from ultralytics import YOLO

# Carrega modelo pré-treinado leve para detecção rápida
modelo = YOLO("yolov8n.pt")

PASTA_ORIGEM = "imagens_brutas"
PASTA_DESTINO = "imagens_cropadas"
PADDING_PERCENTUAL = 0.15  # 15% de margem ao redor para não cortar ponta de asa/rabo
LIMIAR_CONFIANCA = 0.40  # Confiança mínima da detecção
PROPORCAO_MINIMA_AREA = 0.05  # Descarta se a coruja ocupar menos de 5% da foto total

os.makedirs(PASTA_DESTINO, exist_ok=True)

for arquivo in os.listdir(PASTA_ORIGEM):
    if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    caminho_img = os.path.join(PASTA_ORIGEM, arquivo)
    img = cv2.imread(caminho_img)
    if img is None:
        continue

    alt_total, larg_total = img.shape[:2]
    area_total = alt_total * larg_total

    # Roda a inferência (classe 14 no COCO corresponde a 'bird')
    predicoes = modelo(caminho_img, verbose=False)[0]

    boxes_aves = []
    for box in predicoes.boxes:
        if (
            int(box.cls[0]) == 14
            and float(box.conf[0]) >= LIMIAR_CONFIANCA
        ):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes_aves.append((x1, y1, x2, y2))

    # Pula imagens onde nenhuma ave foi encontrada
    if not boxes_aves:
        continue

    # Calcula a caixa delimitadora unificada cobrindo todas as aves encontradas
    min_x = min([b[0] for b in boxes_aves])
    min_y = min([b[1] for b in boxes_aves])
    max_x = max([b[2] for b in boxes_aves])
    max_y = max([b[3] for b in boxes_aves])

    largura_crop = max_x - min_x
    altura_crop = max_y - min_y
    area_crop = largura_crop * altura_crop

    # Descarta fotos onde o animal é minúsculo demais para ser útil
    if (area_crop / area_total) < PROPORCAO_MINIMA_AREA:
        continue

    # Aplica margem (padding) ao redor do corpo para manter o enquadramento natural
    pad_w = int(largura_crop * PADDING_PERCENTUAL)
    pad_h = int(altura_crop * PADDING_PERCENTUAL)

    x_inicial = max(0, min_x - pad_w)
    y_inicial = max(0, min_y - pad_h)
    x_final = min(larg_total, max_x + pad_w)
    y_final = min(alt_total, max_y + pad_h)

    # Executa o crop e salva a imagem limpa
    img_cortada = img[y_inicial:y_final, x_inicial:x_final]
    cv2.imwrite(os.path.join(PASTA_DESTINO, arquivo), img_cortada)