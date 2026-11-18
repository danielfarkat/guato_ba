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

**Baixar os dados** (DOI **[10.5281/zenodo.24292984](https://doi.org/10.5281/zenodo.24292984)**) e colocar os
`.npy` em **`data/`** na raiz do projeto:

```
guato_ba/data/
  ├─ train_v123_256_{T1,T2,Y,G,POS,SPLIT}.npy   # Dataset A
  └─ test2025_{T1,T2,Y,G}.npy                    # Dataset B
```

**Carregar é só numpy:**
```python
import numpy as np
T1 = np.load("data/train_v123_256_T1.npy", mmap_mode="r")   # (13004,4,256,256)
Y  = np.load("data/train_v123_256_Y.npy",  mmap_mode="r")    # {0,1,255}
x  = T1[0].astype("float32") / 10000                         # reflectância
```

> O notebook **[`01_datasets_A_e_B`](notebooks/01_datasets_A_e_B.ipynb)** carrega e visualiza os dois datasets.

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
T2 + `dNBR`, `dNBRSWIR`, `dMIRBI` (deltas t1−t2): `rf`, `xgboost`, `mlp`, `svm` (hiperparâmetros fixos em
`machine_learning/<m>/hpo_best.json`, carregados direto no `train-ml`).

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

# 0) baixe os dados do Zenodo (DOI 10.5281/zenodo.24292984) e coloque os .npy em data/

$P main.py train-ml                       # 1) treina os 4 ML (rf, xgboost, mlp, svm) — lê data/
$P main.py train-dl --modelo unet         # 2) treina um modelo DL (omita --modelo p/ todos)
$P main.py evaluate                        # 3) comparação ML × DL (val 2024 + OOT 2025)
$P main.py accumulate-full --modelo biatt_unet --thr 0.90 --mmu 24  # 4) acumulado 2024
```

Modelos DL: `unet, unet_nored, att_unet, biunet, biatt_unet, resnet34_siam, deeplabv3plus`.
Os passos **1–3** usam apenas os dados do Zenodo (em `data/`). O passo **4** (acumulado anual) acessa o
**Brazil Data Cube** (rede) e o arquivo do bioma (guardrail) — é **retomável** (um `.npz` por tile).
Dependências em [`requirements.txt`](requirements.txt).

---

## 🗂️ Estrutura do repositório

```
guato_ba/
  # --- código núcleo (tudo o que o pipeline precisa) ---
  main.py                CLI por estágio: data · train-ml · train-dl · evaluate · accumulate(-full)
  redes.py               arquiteturas DL (7) + loss + fábricas ML (4)
  geo_data_science.py    bioma/grid 5880, STAC BDC, dataset de patches, features, métricas, inferência
  accumulate_tile.py     acumulado anual 2024 tile-a-tile (Sentinel)

  notebooks/             00..06 — a série explicativa (ver tabela acima)
  machine_learning/<m>/  hpo_best.json — melhores hiperparâmetros dos 4 modelos ML
  results/artigo/        as 2 tabelas de resultado exibidas nos notebooks

  requirements.txt  README.md  DATA.md  .gitignore

  data/                  (você cria) os .npy baixados do Zenodo
```

> Este é um **repositório mínimo e reprodutível**: só o necessário para, com os dados do Zenodo, refazer a
> comparação ML × DL e o acumulado 2024. Os dados pesados e o que não é essencial (pesos, rasters, logs,
> estudos de pesquisa) **não** entram no git — ver **[DATA.md](DATA.md)**.

---

## 📜 Licença e citação

Em homenagem ao povo **Guató**. Ao usar os datasets ou o código, cite o registro do **Zenodo**
(DOI **[10.5281/zenodo.24292984](https://doi.org/10.5281/zenodo.24292984)**) e este repositório:
`https://github.com/danielfarkat/guato_ba`.
