# Dataset A — Treino (`train_v123_256`)
### guató_ba · patches bitemporais Sentinel-2 para área queimada no Pantanal

Patches **256×256** (20 m/pixel, EPSG:5880) recortados do **bioma Pantanal**, cada um um **par
bitemporal** (duas datas, `T1` → `T2`) de Sentinel-2 L2A. Usado para **treinar e validar** os modelos
(ano de referência **2024**, três pares de datas).

> Dataset irmão: **B — Teste OOT 2025** (`test2025`), para avaliação fora da amostra.
> Código, notebooks explicativos e protocolo: https://github.com/danielfarkat/guato_ba

## Arquivos

| arquivo | shape | dtype | conteúdo |
|---|---|---|---|
| `train_v123_256_T1.npy` | (13004, 4, 256, 256) | `uint16` | data **T1** (antes) — 4 bandas |
| `train_v123_256_T2.npy` | (13004, 4, 256, 256) | `uint16` | data **T2** (depois) — 4 bandas |
| `train_v123_256_Y.npy`  | (13004, 256, 256) | `uint8` | rótulo: `0`=não-queimada, `1`=queimada, `255`=inválido |
| `train_v123_256_G.npy`  | (13004,) | `<U4` | **tile** MGRS de origem (grupo p/ GroupKFold) |
| `train_v123_256_POS.npy`| (13004, 3) | `object` | `[par, linha, coluna]` de origem (par ∈ {1,2,3}) |
| `train_v123_256_SPLIT.npy`| (13004,) | `int` | `0`=treino, `1`=validação (split 70/30 do estudo) |

**13.004 patches.**

## Bandas (eixo 1 de T1/T2)

| índice | banda | nome | ~λ |
|---|---|---|---|
| 0 | **B04** | Vermelho | 665 nm |
| 1 | **B8A** | NIR (red-edge) | 865 nm |
| 2 | **B11** | SWIR1 | 1610 nm |
| 3 | **B12** | SWIR2 | 2190 nm |

Valores são **DN** (`uint16`). **Reflectância = DN / 10000**. As entradas já vêm **mascaradas para
nuvem e água** (SCL ∈ {4,5,7} + índice WIW); pixels inválidos recebem `255` no rótulo e devem ser
**ignorados** na perda e nas métricas.

## Como carregar

```python
import numpy as np

T1 = np.load("train_v123_256_T1.npy", mmap_mode="r")   # (13004,4,256,256) uint16
T2 = np.load("train_v123_256_T2.npy", mmap_mode="r")
Y  = np.load("train_v123_256_Y.npy",  mmap_mode="r")    # {0,1,255}
G  = np.load("train_v123_256_G.npy",     allow_pickle=True)   # tile por patch
SPLIT = np.load("train_v123_256_SPLIT.npy", allow_pickle=True)  # 0=treino, 1=val

i = 0
x = np.concatenate([T1[i], T2[i]], 0).astype("float32") / 10000  # (8,256,256) reflectância
y = np.asarray(Y[i])                                             # (256,256)
valido = (y != 255)                                              # use só estes pixels
```

> O notebook **`01_datasets_A_e_B.ipynb`** (no GitHub) mostra tudo isto, com visualização dos patches.

## Métrica do estudo

**IoU da classe queimada** = TP / (TP + FP + FN). Nunca acurácia global nem mIoU (dominadas pela
classe não-queimada). Avaliação sempre **dentro do bioma Pantanal**.

## Citação

Veja o `CITATION`/DOI no registro Zenodo e no repositório. Em homenagem ao povo **Guató**.
