# 🔥 guató_ba — mapeamento automático de áreas queimadas do Pantanal

> *Em homenagem ao povo **Guató**.* Habitantes ancestrais do Pantanal, os Guató foram quase dizimados,
> mas retomaram sua história e seu território. Este estudo nasce do mesmo propósito: **manter a história
> clara**. Mapear automaticamente, ano após ano, as áreas queimadas do Pantanal é registrar o que
> aconteceu com essa terra — para que não se apague.

---

## O que é este projeto

**guató_ba** detecta **cicatrizes de queimada** no Pantanal a partir de pares de imagens **Sentinel-2**
de duas datas (*bitemporal*). Uma cicatriz é, essencialmente, uma **mudança no tempo**: a mesma área
perde vigor de vegetação entre uma data **T1** (antes) e uma data **T2** (depois). O estudo aprende essa
mudança e compara, de forma ampla e justa, **aprendizado profundo (DL)** e **aprendizado de máquina (ML)**,
treinando em **2024** e testando **fora da amostra (OOT)** em **2025** — um ano nunca visto.

Este repositório contém o **código do pipeline**, os **notebooks explicativos** e os **metadados leves**
do estudo. Os dados pesados (patches, rasters, pesos treinados) ficam **fora do git** e são publicados à
parte — ver **[DATA.md](DATA.md)**.

---

## 📦 Os dois datasets (publicados no Zenodo)

O coração do estudo são **dois conjuntos de patches** `256×256` (20 m/pixel, EPSG:5880), recortados do
**bioma Pantanal**. Cada patch é um **par bitemporal** — a mesma área em T1 e T2.

| | Dataset | Arquivos | Patches | Papel |
|---|---|---|---|---|
| **A** | **Treino** — `train_v123_256` | `T1, T2, Y, G, POS, SPLIT` | **13.004** | treinar/validar (2024) |
| **B** | **Teste OOT** — `test2025` | `T1, T2, Y, G` | **710** | testar fora da amostra (2025) |

- **`T1`, `T2`** — `(N, 4, 256, 256)` `uint16`. **4 bandas** nesta ordem: **B04** (vermelho), **B8A** (NIR),
  **B11** (SWIR1), **B12** (SWIR2). Valores são **DN**; **reflectância = DN / 10000**.
- **`Y`** — `(N, 256, 256)` `uint8`: **`0`** não-queimada · **`1`** queimada · **`255`** inválido (nuvem/água/
  fora do bioma) → **sempre ignorado** na perda e nas métricas.
- **`G`** — tile MGRS de origem (grupo para `GroupKFold`, anti-vazamento espacial).
- **`POS`** — `[par, linha, coluna]` de origem (par ∈ {1,2,3} = os três pares de datas de 2024).
- **`SPLIT`** — `0` treino, `1` validação (split estratificado 70/30).

As entradas **já vêm mascaradas** para nuvem e água (SCL ∈ {4,5,7} + índice de água **WIW**).

**Carregar é só numpy:**
```python
import numpy as np
T1 = np.load("train_v123_256_T1.npy", mmap_mode="r")      # (13004,4,256,256)
Y  = np.load("train_v123_256_Y.npy",  mmap_mode="r")       # {0,1,255}
x  = T1[0].astype("float32") / 10000                       # reflectância
```

> READMEs prontos para acompanhar cada zip do Zenodo: **[`zenodo/`](zenodo/)**.
> DOI do Zenodo: _(a preencher após a publicação)_.

---

## 📓 Notebooks — o estudo explicado passo a passo

Comece pelo **00** e siga a numeração. Os notebooks **01–04** estão **executados** (com saídas reais);
o **05** é descritivo (precisa de BDC + GPU) e o **06** é um tutorial reproduzível.

| # | Notebook | O que explica |
|---|---|---|
| 00 | [`00_visao_geral`](notebooks/00_visao_geral.ipynb) | o estudo em resumo + índice + resultados |
| 01 | [`01_datasets_A_e_B`](notebooks/01_datasets_A_e_B.ipynb) | **os datasets A e B**: carregar, entender e visualizar os patches |
| 02 | [`02_modelos_deep_learning`](notebooks/02_modelos_deep_learning.ipynb) | as 7 arquiteturas DL + a função de perda |
| 03 | [`03_modelos_machine_learning`](notebooks/03_modelos_machine_learning.ipynb) | os 4 modelos ML por pixel + os 11 atributos |
| 04 | [`04_metricas_e_avaliacao`](notebooks/04_metricas_e_avaliacao.ipynb) | por que **IoU da queimada** + resultados (val 2024 + OOT 2025) |
| 05 | [`05_acumulado_anual`](notebooks/05_acumulado_anual.ipynb) | como vira **produto anual** do Pantanal (tile-a-tile) |
| 06 | [`06_tutorial_ponta_a_ponta`](notebooks/06_tutorial_ponta_a_ponta.ipynb) | do Brazil Data Cube à cicatriz, reproduzível |
| 07 | [`07_hpo_optuna_neptune`](notebooks/07_hpo_optuna_neptune.ipynb) | busca de hiperparâmetros (Optuna) |

---

## 🧱 A regra inegociável — o guardrail do Pantanal

**Toda** etapa (referência, treino, validação, teste, acumulado, vetores e métricas) é **recortada ao
bioma Pantanal** (`pantanal_mgrs_v2.gpkg`, EPSG:5880 / 20 m) e **nunca** considera nada fora dele
(`mascara_bioma()` em `geo_data_science.py`). As entradas já vêm **sem água e sem nuvem** (SCL ∈ {4,5,7}
+ WIW); pixels inválidos são **zerados após o z-score**, idêntico em treino e inferência. O modelo
**nunca vê** nuvem nem água.

---

## 🧠 Os modelos comparados

**Deep learning (7)** — segmentação bitemporal (entrada = par T1→T2), saída logits `(B,H,W)`:

| modelo | arquitetura | bandas | in_ch | params |
|---|---|---|---|---|
| `unet` | U-Net | red, b8a, b11, b12 | 8 | 17,5 M |
| `unet_nored` | U-Net (sem red) | b8a, b11, b12 | 6 | 17,5 M |
| `att_unet` | U-Net + atenção (Oktay) | red, b8a, b11, b12 | 8 | 18,1 M |
| `deeplabv3plus` | DeepLabV3+ (ASPP) | red, b8a, b11, b12 | 8 | 19,1 M |
| `biunet` | U-Net siamesa | b8a, b11, b12 | 6 | 20,4 M |
| `biatt_unet` | Bi-Attention U-Net (Bi-RAUnet) | b8a, b11, b12 | 6 | 22,0 M |
| `resnet34_siam` | ResNet34-Siamês (ImageNet) | b8a, b11, b12 | 6 | 26,3 M |

*in_ch = bandas × 2 datas.* **Perda:** Focal Tversky (dominante, penaliza omissão) + BCE ponderada por
borda. **Otimização:** AdamW (lr 3e-4), Cosine warm restarts, AMP, dropout 0,2; z-score por canal
estimado só no treino; batch estratificado de 280 patches; early-stop pelo **IoU de validação**.

**Machine learning (4)** — classificação **por pixel**, 11 atributos = reflectância das 4 bandas em T1 e
T2 + `dNBR`, `dNBRSWIR`, `dMIRBI` (deltas t1−t2): `rf`, `xgboost`, `mlp`, `svm` (hiperparâmetros por
Optuna em `machine_learning/<m>/hpo_best.json`).

---

## 📊 A métrica e os resultados

Métrica central = **IoU da classe queimada** = TP/(TP+FP+FN) (= CSI/Jaccard da cicatriz), com
**OE (omissão)** = FN/(TP+FN) e **CE (comissão)** = FP/(TP+FP). **Nunca** mIoU nem acurácia global
(dominadas pela classe não-queimada). Teste honesto = **OOT 2025**:

| modelo | tipo | IoU (OOT 2025) | omissão | comissão |
|---|---|---|---|---|
| `unet` | DL | **0,745** | 0,222 | 0,044 |
| `deeplabv3plus` | DL | 0,742 | 0,224 | 0,045 |
| `att_unet` | DL | 0,730 | 0,236 | 0,047 |
| `rf` | ML | 0,522 | 0,181 | 0,570 |
| `xgboost` | ML | 0,519 | 0,179 | 0,582 |

*(tabela completa em `results/artigo/modelos_ml_dl_desempenho.csv` e no notebook 04.)* As U-Nets
generalizam melhor para o ano nunca visto; os modelos por pixel detectam bem, mas **superestimam** a área
(comissão alta) por não usarem contexto espacial.

---

## ▶️ Como rodar o pipeline

Sempre com o interpretador **geospatial** (torch CUDA + rasterio/geopandas):

```bash
cd guato_ba
P=/opt/conda/envs/geospatial/bin/python

$P main.py reference                      # 1) valida o guardrail (referência ∩ bioma, 0 pixel fora)
$P main.py data --forcar                  # 2) (re)constrói os patches 2024 (3 pares) + teste OOT 2025
$P main.py train-dl --modelo unet         # 3) treina um modelo DL (70/30, batch 280, early-stop)
$P main.py hpo-ml && $P main.py train-ml  # 4) ML: busca Optuna + treino (rf, xgboost, mlp, svm)
$P main.py evaluate                        # 5) avalia val 2024 + OOT 2025 -> results/tabela_patch.csv
$P main.py accumulate-full --modelo biatt_unet --cloud 10 --in_bioma 0.50 --thr 0.90 --mmu 24  # 6) produto anual
```

Modelos DL: `unet, unet_nored, att_unet, biunet, biatt_unet, resnet34_siam, deeplabv3plus`.
O acumulado é **retomável** (grava um `.npz` por tile em `results/acumulado/_ckpt_*/`).
Dependências em [`requirements.txt`](requirements.txt).

---

## 🗂️ Estrutura do repositório

```
guato_ba/
  # --- núcleo (importável; fica na raiz) ---
  main.py                fluxo (CLI por estágio) + treino_dl + avaliação
  redes.py               arquiteturas DL + loss + fábricas ML + espaços Optuna
  geo_data_science.py    bioma/grid 5880, STAC BDC, dataset de patches, features, métricas, inferência
  accumulate_tile.py     acumulado 2024 tile-a-tile (Sentinel)
  accumulate_landsat.py  acumulado Landsat (transfer da DL 3-bandas)

  notebooks/             00..07 — a série explicativa (ver tabela acima)
  zenodo/                READMEs dos datasets A e B (acompanham os zips do Zenodo)
  scripts/               experimentos: split/ acumulado/ ablacao/ eval/ figuras/
  logs/                  *_resultados.log (logs de execução dos estudos)

  # --- metadados LEVES do estudo (versionados) ---
  deeplearning/<m>/      historico.csv, curva.png, hpo_best.json   (pesos *.pkl ficam FORA — ver DATA.md)
  machine_learning/<m>/  hpo_best.json
  hpo/                   *.db (estudos Optuna, reproduzíveis)
  results/*.csv          tabela_patch.csv, results/artigo/*.csv

  README.md  DATA.md  requirements.txt  .gitignore
```

> Os dados pesados (`data/`, `dados/`, rasters de `results/`, pesos `*.pkl`, `artigo/`) **não** entram no
> git. O que fica fora e como obter: **[DATA.md](DATA.md)**.

---

## 📜 Licença e citação

Em homenagem ao povo **Guató**. Ao usar os datasets ou o código, cite o registro do **Zenodo**
(DOI a preencher) e este repositório: `https://github.com/danielfarkat/guato_ba`.
