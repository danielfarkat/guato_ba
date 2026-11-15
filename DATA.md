# Dados — o que fica fora do repositório

Este repositório versiona **código** e **metadados leves** (históricos de treino, curvas,
melhores hiperparâmetros, tabelas, estudos Optuna). Os dados **pesados** e os **pesos treinados**
**não** são versionados — ficam no disco de trabalho (e abertos no QGIS) e são publicados à parte.

## Não versionado (e por quê)

| Caminho | Tam. aprox. | Conteúdo | Como obter |
|---|---|---|---|
| `data/` | ~15 GB | caches `.npy` dos patches (`train_v123_256_*`, `test2025_*`) | Zenodo (abaixo) ou regerar: `main.py data --forcar` |
| `dados/` | ~4.8 GB | rasters de entrada (`referencia_hls*.tif`, cubo `21kwb/`) | insumos locais do estudo |
| `results/` (rasters) | ~7.7 GB | acumulados, máscaras, matriz RGB, predições 2025 (`.tif`/`.gpkg`/`.npz`) | regerável pelo pipeline (`accumulate*`) |
| `deeplearning/<m>/*.pkl` | ~3.3 GB | pesos DL treinados (`*_best.pkl`, `*_last.pkl`) | retreinar (`main.py train-dl`) |
| `machine_learning/<m>/*.pkl` | ~91 MB | modelos ML treinados | retreinar (`main.py train-ml`) |
| `artigo/` | ~23 MB | dissertação (LaTeX, PDF, zips Overleaf) | material do artigo |

> As **referências fixas** do estudo (`pantanal_mgrs_v2.gpkg`, máscaras curadas, HLS/DETER/MapBiomas,
> cubos `bdc_outputs/…`) ficam **fora** de `guato_ba/` e **nunca** são regeradas — apenas carregadas.
> Os caminhos estão centralizados em `geo_data_science.py` (`RAIZ`, `GUATO`, `DADOS`, `DATA_DIR`, ...).

## Datasets publicados (Zenodo)

Os patches de treino e teste são publicados como dois zips (cada um com um `README.md` de shapes,
dtypes, ordem de bandas e uso):

- **Dataset 1 — Treino** (`train_v123_256`): 13.004 patches 256×256, pares bitemporais Sentinel-2
  (`T1`, `T2` em `uint16`, 4 bandas `[B04, B8A, B11, B12]`), máscara `Y`, grupo por tile `G` (GroupKFold),
  `POS`, `SPLIT`.
- **Dataset 2 — Teste 2025 (OOT)** (`test2025`): 710 patches 256×256, mesma estrutura.

Bandas: Sentinel-2 L2A, **reflectância = DN / 10000**. Entradas já mascaradas para água e nuvem.
Métrica do estudo: **IoU da classe queimada** (TP/(TP+FP+FN)).

> DOI do Zenodo: _(a preencher após a publicação do registro)_
