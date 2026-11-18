# Dados — o que fica fora do repositório

Este repositório versiona **só o mínimo** para reproduzir: o código núcleo, os notebooks explicativos,
os **hiperparâmetros dos modelos ML** (`machine_learning/<m>/hpo_best.json`) e as **2 tabelas** de
resultado exibidas nos notebooks. Os dados **pesados**, os **pesos treinados** e os artefatos de pesquisa
(logs, scripts, históricos/curvas de treino, acumulados) **não** são versionados — ficam no disco de
trabalho (e abertos no QGIS) e, no caso dos datasets, são publicados no Zenodo.

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

Os patches de treino e teste são publicados como dois zips. O `README.md` de cada zip (shapes, dtypes,
ordem de bandas e uso) já está versionado em **[`zenodo/`](zenodo/)** e o notebook
**[`notebooks/01_datasets_A_e_B.ipynb`](notebooks/01_datasets_A_e_B.ipynb)** mostra como carregar e
visualizar os dois:

- **Dataset A — Treino** (`train_v123_256`): 13.004 patches 256×256, pares bitemporais Sentinel-2
  (`T1`, `T2` em `uint16`, 4 bandas `[B04, B8A, B11, B12]`), máscara `Y`, grupo por tile `G` (GroupKFold),
  `POS`, `SPLIT`. → `zenodo/README_dataset_A_treino.md`
- **Dataset B — Teste 2025 (OOT)** (`test2025`): 710 patches 256×256, mesma estrutura.
  → `zenodo/README_dataset_B_teste2025.md`

Bandas: Sentinel-2 L2A, **reflectância = DN / 10000**. Entradas já mascaradas para água e nuvem.
Rótulo `Y`: `0`=não-queimada, `1`=queimada, `255`=inválido (ignorar).
Métrica do estudo: **IoU da classe queimada** (TP/(TP+FP+FN)).

> DOI do Zenodo: _(a preencher após a publicação do registro)_
