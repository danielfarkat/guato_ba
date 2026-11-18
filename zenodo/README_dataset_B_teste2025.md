# Dataset B — Teste OOT 2025 (`test2025`)
### guató_ba · patches bitemporais Sentinel-2 para área queimada no Pantanal

Patches **256×256** (20 m/pixel, EPSG:5880) do **bioma Pantanal**, cada um um **par bitemporal**
(`T1` → `T2`) de Sentinel-2 L2A do ano **2025**. Serve para **teste fora da amostra (OOT)**: um ano
que os modelos **nunca viram** no treino — a avaliação mais honesta de generalização.

> Dataset irmão: **A — Treino** (`train_v123_256`), com 13.004 patches de 2024.
> Código, notebooks explicativos e protocolo: https://github.com/danielfarkat/guato_ba

## Arquivos

| arquivo | shape | dtype | conteúdo |
|---|---|---|---|
| `test2025_T1.npy` | (710, 4, 256, 256) | `uint16` | data **T1** (antes) — 4 bandas |
| `test2025_T2.npy` | (710, 4, 256, 256) | `uint16` | data **T2** (depois) — 4 bandas |
| `test2025_Y.npy`  | (710, 256, 256) | `uint8` | rótulo: `0`=não-queimada, `1`=queimada, `255`=inválido |
| `test2025_G.npy`  | (710,) | `<U3` | **tile** MGRS de origem |

**710 patches.** Mesma estrutura e convenções do Dataset A.

## Bandas, reflectância e rótulo

Idênticos ao Dataset A: eixo de bandas `[B04, B8A, B11, B12]`; **reflectância = DN / 10000**; rótulo
`Y ∈ {0, 1, 255}` com `255` = inválido (nuvem/água/fora do bioma, a ignorar). Entradas já mascaradas.

## Como carregar

```python
import numpy as np

T1 = np.load("test2025_T1.npy", mmap_mode="r")   # (710,4,256,256) uint16
T2 = np.load("test2025_T2.npy", mmap_mode="r")
Y  = np.load("test2025_Y.npy",  mmap_mode="r")    # {0,1,255}

i = 0
x = np.concatenate([T1[i], T2[i]], 0).astype("float32") / 10000   # (8,256,256)
y = np.asarray(Y[i]); valido = (y != 255)
```

> O notebook **`01_datasets_A_e_B.ipynb`** (no GitHub) carrega e visualiza A **e** B lado a lado.

## Métrica do estudo

**IoU da classe queimada** = TP / (TP + FP + FN), sempre **dentro do bioma Pantanal**. Em homenagem ao
povo **Guató**.
