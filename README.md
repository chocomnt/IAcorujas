# Classificação de espécies de corujas com redes neurais

Projeto de Visão Computacional para classificar imagens de **20 espécies de corujas**. O repositório documenta a evolução de uma CNN própria treinada do zero e um experimento separado com MobileNetV2 pré-treinada na ImageNet.

> **Resumo:** a melhor versão da CNN própria foi a V4, com 58,63% de acurácia e 22,82% de F1 macro no teste. O experimento MobileNetV2 atingiu 82,66% de acurácia e 66,28% de F1 macro no mesmo split de teste, mostrando o ganho do transfer learning nesta base.

## Abrir no Google Colab

Os notebooks foram preparados para execução no Google Colab. Para treinamento, selecione **Ambiente de execução → Alterar tipo de ambiente de execução → GPU**.

| Notebook | Descrição | Colab |
|---|---|---|
| `IACorujaV1.ipynb` | CNN inicial — versão base | [Abrir no Colab](https://colab.research.google.com/github/chocomnt/IAcorujas/blob/master/IACorujaV1.ipynb) |
| `IACorujaV2.ipynb` | CNN própria revisada, divisão treino/validação/teste e métricas macro | [Abrir no Colab](https://colab.research.google.com/github/chocomnt/IAcorujas/blob/master/IACorujaV2.ipynb) |
| `IACorujaV3.ipynb` | Experimento com limite de 400 imagens por classe no treino | [Abrir no Colab](https://colab.research.google.com/github/chocomnt/IAcorujas/blob/master/IACorujaV3.ipynb) |
| `IACorujaV4.ipynb` | Base inteira e pesos de classe intermediários; melhor CNN própria | [Abrir no Colab](https://colab.research.google.com/github/chocomnt/IAcorujas/blob/master/IACorujaV4.ipynb) |
| `IACorujaV5.ipynb` | Busca automática de pesos e ajuste pós-treino de vieses | [Abrir no Colab](https://colab.research.google.com/github/chocomnt/IAcorujas/blob/master/IACorujaV5.ipynb) |
| `IAcorujaMobileNetV2.ipynb` | Experimento separado com transfer learning | [Abrir no Colab](https://colab.research.google.com/github/chocomnt/IAcorujas/blob/master/IAcorujaMobileNetV2.ipynb) |
| `Apresentacao_Final_Corujas.ipynb` | Apresentação consolidada dos experimentos e resultados | [Abrir no Colab](https://colab.research.google.com/github/chocomnt/IAcorujas/blob/master/Apresentacao_Final_Corujas.ipynb) |

## Problema e base de dados

O problema é uma **classificação multiclasse**: dada uma imagem, prever qual das 20 espécies ela mostra. As imagens vêm do iNaturalist e, nos experimentos registrados, o arquivo de imagens continha **8.067 imagens**.

A base é fortemente desbalanceada:

- `Athene_cunicularia`: 4.415 imagens;
- `Megascops_watsonii`: 26 imagens;
- razão entre maior e menor classe: aproximadamente 170:1.

O arquivo compactado utilizado pelos notebooks é baixado de um Google Drive compartilhado por `gdown`. Se o arquivo deixar de estar acessível, será necessário atualizar o ID de download nas células de preparação dos dados. O diretório local `images/` é ignorado pelo Git e não precisa ser versionado para executar os notebooks que fazem o download do arquivo.

> **Licenças:** as imagens são originárias do iNaturalist e podem ter licenças diferentes. Verifique a licença de cada imagem antes de reutilizá-la ou redistribuí-la fora do contexto do projeto.

## Metodologia

### CNN própria

A partir da V2, os experimentos usam divisão estratificada de **70% treino, 15% validação e 15% teste**, com semente 42. O teste é reservado para a avaliação final. A V1 é uma exceção: usa 80% treino e 20% validação, sem teste separado; por isso seus resultados não são diretamente comparáveis aos posteriores.

A V4, melhor CNN própria, tem:

- entrada de 224×224×3;
- data augmentation com flip horizontal, rotação, zoom e contraste;
- normalização de pixels para [0, 1];
- quatro blocos convolucionais, com filtros 32, 64, 128 e 256;
- duas convoluções 3×3 por bloco, Batch Normalization, ReLU, MaxPooling e Dropout;
- Global Average Pooling, Dense de 128 neurônios e saída Softmax de 20 classes;
- 1.210.612 parâmetros, aproximadamente 4,62 MB;
- Adam, sparse categorical crossentropy e batch size 32.

Os pesos de classe da V4 foram calculados no conjunto de treino com pesos balanceados suavizados por `alpha=0.75`, limitados entre 0,15 e 3,0 e normalizados. A seleção do melhor ponto de treino usa F1 macro na validação.

### Métricas

Além da acurácia global, o projeto usa acurácia balanceada, precisão/recall/F1 macro, relatório por espécie, top-3 e matriz de confusão normalizada. Essas métricas são importantes porque a acurácia global pode ser dominada pela classe mais frequente.

## Resultados registrados

| Experimento | Acurácia | Acc. balanceada | Precisão macro | Recall macro | F1 macro | Top-3 | Classes com recall zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 — validação 80/20 | 62,49% | 10,81% | 10,34% | 10,81% | 9,55% | 77,12% | 14 |
| V2 — teste 70/15/15 | 60,03% | 13,38% | 14,13% | 13,38% | 12,66% | — | 12 |
| V3 — teste 70/15/15 | 31,46% | 20,67% | 12,52% | 20,67% | 11,74% | — | 6 |
| **V4 — teste 70/15/15** | **58,63%** | **25,78%** | **23,82%** | **25,78%** | **22,82%** | — | **3** |
| V5 — teste 70/15/15 | 58,55% | 18,06% | 18,17% | 18,06% | 17,08% | — | 8 |
| **MobileNetV2 — teste 70/15/15** | **82,66%** | **70,43%** | **66,51%** | **70,43%** | **66,28%** | **95,38%** | **0** |

**Atenção à comparação:** V1 foi avaliada em validação 80/20, não em teste. V2–V5 e MobileNetV2 usam o split 70/15/15. A comparação mais direta é V4 versus MobileNetV2, pois ambas foram avaliadas no mesmo conjunto de teste.

### O que os experimentos mostram

- A V1 tinha acurácia global razoável, mas F1 macro baixo: a rede favorecia as classes frequentes.
- A V3 mostrou que limitar agressivamente a classe majoritária e usar pesos completos pode elevar o recall de classes raras, mas também gerar muitos falsos positivos e derrubar a acurácia.
- A V4 encontrou o melhor equilíbrio da CNN própria: quase manteve a acurácia da V2 e elevou o F1 macro de 12,66% para 22,82%.
- A V5 tentou automatizar a busca dos pesos e ajustar vieses por classe. O ajuste melhorou as métricas na validação, mas piorou no teste, indicando sobreajuste à validação.
- A MobileNetV2 foi muito superior neste conjunto, tanto em acurácia como em métricas macro, e não teve classes com recall zero.

## MobileNetV2: experimento de transfer learning

MobileNetV2 é uma arquitetura CNN eficiente, com convoluções *depthwise separable*. No projeto, foi carregada com pesos pré-treinados na ImageNet, sua cabeça original foi removida e uma nova saída de 20 classes foi treinada. O treinamento ocorreu em duas etapas: primeiro com o backbone congelado e depois com fine-tuning das camadas finais.

O resultado de 82,66% de acurácia e 66,28% de F1 macro mostra o potencial de reutilizar características aprendidas em uma base grande. A MobileNetV2 é um **experimento separado derivado da V2**, não uma versão da CNN própria nem o modelo principal exigido como arquitetura construída no projeto.

## Como executar

1. Abra o notebook desejado no Colab pelo link acima.
2. Para treinamento, selecione uma GPU (T4 é suficiente; L4/A100 também funcionam).
3. Execute as células em ordem.
4. Na primeira execução, o notebook baixa e extrai a base no armazenamento temporário `/content`.
5. Acompanhe as métricas e os gráficos no próprio notebook.

O armazenamento do Colab é temporário. Ao encerrar ou reiniciar a sessão, arquivos em `/content` e modelos que não foram explicitamente salvos deixam de estar disponíveis. A apresentação consolidada não treina modelos; ela exibe os resultados registrados.

## Estrutura do repositório

```text
.
├── Apresentacao_Final_Corujas.ipynb
├── IACorujaV1.ipynb
├── IACorujaV2.ipynb
├── IACorujaV3.ipynb
├── IACorujaV4.ipynb
├── IACorujaV5.ipynb
├── IAcorujaMobileNetV2.ipynb
├── datasetStrigidae_filtered.csv
├── download_images.py
├── matriz_confusao_v4.png
└── yolocut.py
```

## Limitações

- Algumas espécies têm poucas imagens, tornando o recall instável.
- O teste não substitui validação externa em câmeras de campo.
- O classificador sempre escolhe uma das 20 espécies; não tem classe “desconhecida”.
- A CNN própria é uma prova de conceito. A MobileNetV2 demonstra que transfer learning pode elevar significativamente o desempenho.
- Antes de uso real, seria necessário ampliar e revisar os dados, testar em imagens de campo e medir latência e consumo em hardware-alvo.

## Autores

Matheus e Kauan — projeto acadêmico de Visão Computacional e Redes Neurais.
