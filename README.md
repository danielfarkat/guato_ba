# guató_ba — mapeamento automático de áreas queimadas do Pantanal

> *Em homenagem ao povo **Guató**.* Habitantes ancestrais do Pantanal, os Guató foram
> quase dizimados, mas retomaram sua história e seu território. Este estudo nasce do mesmo
> propósito: **manter a história clara**. Mapear automaticamente, ano após ano, as áreas
> queimadas do Pantanal é registrar o que aconteceu com essa terra — para que não se apague.

## Objetivo

Comparar, de forma ampla e justa, arquiteturas de **aprendizado profundo (DL)** e **aprendizado de
máquina (ML)** para detecção de cicatriz de queimada no Pantanal a partir de pares bitemporais
Sentinel-2, treinando em 2024 e testando fora de amostra (OOT) em 2025.

## GUARDRAIL #1 — sempre dentro do Pantanal

**Toda** etapa (referência, treino, validação, teste, acumulado, vetores e métricas) é **recortada ao
bioma Pantanal** (`pantanal_mgrs_v2.gpkg`, EPSG:5880 / 20 m) e **nunca** considera nada fora dele
(`mascara_bioma()` em `geo_data_science.py`). As **entradas já vêm sem água e sem nuvem**: SCL∈{4,5,7}
+ índice de água WIW; pixels inválidos são **zerados após o z-score**, idêntico em treino e inferência.

## Modelos e configurações

**Deep learning (7)** — segmentação bitemporal (entrada = par t1→t2 concatenado), saída logits (B,H,W):

| modelo | arquitetura | bandas usadas | in_ch | micro-batch |
|---|---|---|---|---|
| `unet` | U-Net | red, b8a, b11, b12 | 8 | 256 |
| `unet_nored` | U-Net (sem red) | b8a, b11, b12 | 6 | 256 |
| `att_unet` | U-Net + atenção | red, b8a, b11, b12 | 8 | 128 |
| `biunet` | U-Net siamês | b8a, b11, b12 | 6 | 128 |
| `biatt_unet` | Bi-Attention U-Net (Bi-RAUnet) | b8a, b11, b12 | 6 | 128 |
| `resnet34_siam` | ResNet34-Siamês (ImageNet) | b8a, b11, b12 | 6 | 64 |
| `deeplabv3plus` | DeepLabV3+ (ASPP) | red, b8a, b11, b12 | 8 | 96 |

*in_ch = bandas × 2 datas. O micro-batch alimenta a GPU; o batch efetivo é 280 via acumulação de gradiente.*

**Machine learning (4)** — classificação **por pixel**, 11 atributos = reflectância das 4 bandas em t1 e
t2 + `dNBR`, `dNBRSWIR`, `dMIRBI` (deltas t1−t2):

| modelo | melhores hiperparâmetros (Optuna, IoU de validação) |
|---|---|
| `rf` (Random Forest) | n_estimators=270, max_depth=21, min_samples_leaf=1, max_features=sqrt |
| `xgboost` | n_estimators=400, max_depth=7, lr=0,020, subsample=0,65, colsample=0,68, min_child_weight=3 |
| `mlp` | hidden=(256,128,64), alpha=1,1e-4, lr_init=3,6e-3 |
| `svm` | C=64,7, gamma=scale (RBF) |

## Treinamento (DL)

- **Perda:** Focal Tversky (dominante) + BCE ponderada por borda — `make_loss(border_w=4, tv_a=0.3,
  tv_b=0.7, tv_g=0.75, w_bce=0.5)`.
- **Otimização:** AdamW (lr=3e-4, weight_decay=1e-4), `CosineAnnealingWarmRestarts` (T_0=20·passos/época,
  T_mult=2), AMP (mixed precision), dropout 0,2.
- **Padronização:** z-score por canal, estimado **só no treino** (média/desvio salvos no checkpoint).
- **Augmentation:** espelhamentos (H/V) + rotações 90° por patch.
- **Amostragem (atual):** batch estratificado de **280 patches = ~36 queimada + 244 não-queimada
  (~12,8%)**, **sem reposição** — cada patch (queimada e não-queimada) é visto **1×/época** → cobre todas
  as cicatrizes e áreas. Para o v123 são ~128 batches/época (35.840 patches de treino).
- **Split:** **70% treino / 30% validação, estratificado** (preserva a taxa de queimada); early-stop pelo
  IoU de validação (cap de épocas configurável).
- **Métrica otimizada/monitorada:** **IoU da classe queimada**.

## Dados e protocolo

- **Treino+validação (2024):** patches 128×128, `SCL_MIN=0.30` (≥30% de dado válido no bioma por patch),
  de **3 pares de datas**: 30/05→19/07 (datas_1), 19/07→02/09 (datas_2), 03/08→02/09 (datas_3, curado).
- **Teste OOT (2025):** par 04/09→31/10, recortado ao Pantanal, independente do treino.
- **Referência:** máscaras **fixas, curadas à mão** — *NUNCA regeradas, só carregadas*
  (`carregar_referencia(ano)`): 2024 = `mascara_datas_1/2/3_1_2_2.gpkg`; 2025 = `mascara_datas_1_1_2_2.gpkg`.
- **Acumulado anual:** **tile-a-tile** (sem mosaico global), jan→dez; cenas com nuvem **<10%** → compõe
  revisitas do dia → mantém datas com cobertura válida **≥50%** do tile → pares consecutivos → modelo só em
  pixels válidos → união → clip bioma → MMU 24 px (~1 ha). Reporta a **% do Pantanal efetivamente analisada**.

## Como rodar

Sempre com o interpretador **geospatial** (torch CUDA + rasterio/geopandas):

```bash
cd /home/jovyan/ID-QUEIMADAS/jupyter/guato_ba
P=/opt/conda/envs/geospatial/bin/python

# 1) valida o guardrail (referência ∩ bioma, 0 pixel fora)
$P main.py reference

# 2) (re)constrói os patches de 2024 (3 pares) + teste OOT 2025   [--forcar refaz o cache]
$P main.py data --forcar

# 3) treina um modelo DL (70/30 estratificado, batch 280, early-stop)
$P main.py train-dl --modelo unet --min_stop 40 --patience 20 --epochs 60
#    (modelos: unet, unet_nored, att_unet, biunet, biatt_unet, resnet34_siam, deeplabv3plus)

# 4) ML: busca de hiperparâmetros (Optuna) + treino
$P main.py hpo-ml ; $P main.py train-ml        # rf, xgboost, mlp, svm

# 5) avaliação por patch (validação 2024 + teste OOT 2025) de todos os modelos treinados
$P main.py evaluate                            # -> results/tabela_patch.csv

# 6) acumulado anual 2024
$P main.py accumulate      --modelo biatt_unet --cloud 10 --in_bioma 0.50 --thr 0.90 --mmu 24 --tiles 21KWB,21KVB
$P main.py accumulate-full --modelo biatt_unet --cloud 10 --in_bioma 0.50 --thr 0.90 --mmu 24   # Pantanal inteiro (checkpoint por tile)
$P accumulate_landsat.py   biatt_unet          # acumulado Landsat (transfer da DL 3-bandas)

# 7) estudo comparativo de datasets × modelos (ablação) — treina + acumula 21KWB/21KVB + avalia vs HLS
$P scripts/ablacao/ablacao_datasets.py
```

> Os módulos-núcleo (`main.py`, `redes.py`, `geo_data_science.py`, `accumulate_*.py`) ficam na **raiz**;
> os scripts de experimento foram agrupados em `scripts/<grupo>/` (ver Estrutura). Todos acham o núcleo
> por caminho absoluto, então rodam de qualquer diretório.

**Retomável:** o acumulado (`accumulate-full`/Landsat) grava um `.npz` por tile em
`results/acumulado/_ckpt_<modelo><tag>/`; reiniciar reaproveita os tiles prontos. A ablação pula células já
concluídas (`_last.pkl` + acumulado presentes).

## Métricas

Métrica central = **IoU da classe queimada** = TP/(TP+FP+FN) (equivale ao CSI/Jaccard da cicatriz),
com **OE (omissão)** = FN/(TP+FN) e **CE (comissão)** = FP/(TP+FN); também F1=Dice. **Nunca** mIoU nem
acurácia global (dominadas pela classe não-queimada). Toda avaliação é dentro do bioma; o acumulado reporta
a **% do Pantanal analisada** para comparação justa com referências (HLS, DETER, MapBiomas).

## Estrutura

```
guato_ba/
  # --- núcleo (importável; fica na raiz) ---
  main.py                fluxo (CLI por estágio) + treino_dl + avaliação
  redes.py               arquiteturas DL + loss + fábricas ML + espaços Optuna
  geo_data_science.py    bioma/grid 5880, STAC BDC, dataset de patches, features, métricas, inferência
  accumulate_tile.py     acumulado 2024 tile-a-tile (Sentinel) — run() e run_pantanal()
  accumulate_landsat.py  acumulado Landsat (coleção landsat-2; QA_PIXEL por bits; transfer da DL 3-bandas)

  # --- experimentos / análise (agrupados) ---
  scripts/split/         split espacial honesto norte/sul (anti-vazamento de grupos)
  scripts/acumulado/     acumulados 2023/gsplit, sweep de nuvem, mergulho no tile 21KVT
  scripts/ablacao/       ablação dataset×modelo (OOT 2025), diagnóstico overfitting/GroupNorm
  scripts/eval/          avaliação do acumulado vs referência nova e HLS; combinado S2+Landsat
  scripts/figuras/       figuras/tabelas/produtos p/ artigo e QGIS (mapa_2025, exportar_patches_qgis, ...)
  notebooks/             guato_ba_tutorial.ipynb, hpo_optuna_neptune.ipynb
  logs/                  *_resultados.log (logs de execução dos estudos)

  # --- metadados LEVES do estudo (versionados) ---
  deeplearning/<m>/      historico.csv, curva.png, hpo_best.json   (pesos *.pkl ficam FORA — ver DATA.md)
  machine_learning/<m>/  hpo_best.json                             (*.pkl fora)
  hpo/                   *.db (estudos Optuna, reproduzíveis)
  results/*.csv          tabela_patch.csv, tabela_acumulado_*.csv, results/artigo/*.csv
  reference/             figura de referência

  # --- dados pesados (NÃO versionados; só no disco/QGIS; publicados no Zenodo) ---
  data/   dados/   results/ (rasters)   artigo/   →  ver DATA.md
```

Caminhos e dados fora do repositório: ver **[DATA.md](DATA.md)**.
