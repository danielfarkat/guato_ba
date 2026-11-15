# Proposta de reorganização — `guato_ba`

> **STATUS: PROPOSTA. NADA FOI ALTERADO.** Este documento apenas descreve uma reorganização
> para o usuário aprovar depois. Nenhum arquivo foi movido, renomeado, apagado ou editado.
> O único arquivo criado foi este `.md`.
>
> **Motivo da cautela:** os caminhos estão *hardcoded* nos scripts (quase tudo via a variável
> central `GUATO`, mas há caminhos absolutos também) e os dados são abertos no **QGIS**.
> Já houve incidente de perda por mover/apagar. Qualquer mudança real pode quebrar scripts e QGIS.
> Raiz do projeto analisada: `/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba` (= `GUATO`). Tamanho total: **~35 GB**.
> Ainda **não** é um repositório git (`.git` e `.gitignore` ausentes).

---

## (a) Inventário atual resumido

### Totais por tipo (projeto inteiro)
`.py` 25 · notebooks 2 (+cópias em `.ipynb_checkpoints`) · `.tif` 159 · `.gpkg` 38 · `.npy` 10 ·
`.pkl` 56 · `.csv` 62 · `.png` 47 · `.json` 30 · `.log` 16 · `.db` 5 · `.zip` 6

### Diretórios (por tamanho)

| Dir | Tam. | Conteúdo | Papel |
|---|---|---|---|
| `data/` | 15 G | `train_v123_256_{T1,T2,Y,G,POS,SPLIT}.npy`, `test2025_{T1,T2,Y,G}.npy` | **Cache** do dataset de patches (gerado por `DATA_DIR`). Regenerável. |
| `results/` | 7.7 G | `acumulado/` (2.6 G), `artigo/`, `mascaras_2025/`, `matriz_rgb/`, `patches_qgis/`, `pred2025/`, `qgis_2025/`, `split_espacial/` + soltos | **Saídas** (rasters/vetores/tabelas/figuras). |
| `dados/` | 4.8 G | `referencia_hls.tif` (791 M), `referencia_hls_mes.tif` (8 M, usado por `HLS_MES`), `21kwb/` (~4 G, cubo de cenas do tile 21KWB) | **Entradas** locais (rasters). |
| `deeplearning/` | 3.3 G | 25 subpastas de modelo (`unet`, `biatt_unet`, `att_unet`, `biunet`, `deeplabv3plus`, `resnet34_siam`, variantes `unet_*`), cada uma com `*_best.pkl`/`*_last.pkl` + `curva.png` + `historico.csv` + `hpo_best.json` | **Modelos DL** treinados + metadados. |
| `dados ` ⚠️ | 1.2 G | **pasta com ESPAÇO no fim do nome**: `download?arq=...` (411 M, download cru), `deter_pantanal_shape.zip` (0 byte), `wget-log`, `referencia_hls.tif` (791 M, duplicata), `limites_pantanal.gpkg` (626 K) | **Scratch acidental** de `wget`. Ver nota abaixo. |
| `hpo/` | 336 M | `dl_unet.db`, `ml_{mlp,rf,svm,xgboost}.db` (estudos Optuna) + `_tmp_resnet34_siam/` (202 M), `_tmp_unet/` (134 M) com `*.pkl` | **HPO**: bancos Optuna + checkpoints temporários. |
| `machine_learning/` | 91 M | `mlp/`, `rf/`, `svm/`, `xgboost/`, cada um com `<nome>.pkl` + `hpo_best.json` | **Modelos ML** treinados. |
| `artigo/` | 23 M | `Dissertacao_*_v{5..8}_overleaf.zip`, `Dissertacao_PROVA_conteudo.pdf`, `guato_ba_artigo.{tex,pdf,aux,log,out}`, `compile.log`, `fig_oot2025_iou.png`, `S2-16D-2.2024.06.30.csv`, `Untitled Folder/` (vazia), `.ipynb_checkpoints/` | **Dissertação/artigo** (LaTeX + PDFs + zips Overleaf). |
| `reference/` | 2.5 M | `area_pantanal_0001.png` (+ cópia em checkpoint) | Figura de referência. |
| `notebooks/` | 20 K | `hpo_optuna_neptune.ipynb` | Notebook de HPO. |
| `dissertação/` | vazia | — | Pasta vazia (nome acentuado). Lixo. |
| `__pycache__/`, `.ipynb_checkpoints/` (×6) | 364 K+ | `.pyc` / cópias `-checkpoint` | **Lixo** (regenerável). |

### Scripts Python (25) e papel

**Núcleo / pacote (módulos importados pelos demais — "o motor"):**

| Arquivo | Papel | Importado como |
|---|---|---|
| `geo_data_science.py` | Camada de geociência: **constantes e TODOS os caminhos** (`RAIZ`, `GUATO`, `DADOS`, `DATA_DIR`, refs fixas), grid/bioma, carregamento de referências, construção de dataset, métricas | `g` / `G` |
| `redes.py` | Arquiteturas DL (`construir_dl`, `MODELOS_DL`, `_NORM`) | `from redes import …` |
| `main.py` | Fluxo do estudo por estágio (CLI); define `GUATO = G.GUATO`, `ACU` | `M` |
| `accumulate_tile.py` | Acumulado tile-a-tile (Sentinel); define `ACU_DIR` | `A` / `AT` |
| `accumulate_landsat.py` | Acumulado tile-a-tile (Landsat BDC `landsat-2`); `ACU_DIR = AT.ACU_DIR` | — |

**Scripts de experimento/análise (importam o núcleo; executáveis soltos):**

| Grupo | Arquivos | Descrição |
|---|---|---|
| Split espacial | `split_espacial.py`, `split_espacial_norte.py`, `sim_gsplit.py` | Split espacial honesto norte/sul (anti-vazamento de grupos) |
| Acumulado (anos/variantes) | `acum2023.py`, `acum_gsplit.py`, `sweep2023.py`, `tile21kvt_2023.py` | Acumulados 2023/gsplit, sweep de nuvem, mergulho no tile 21KVT |
| Ablação/diagnóstico | `ablacao_datasets.py`, `diag_5050.py`, `diag_gn.py`, `teste_256.py` | Ablação dataset×modelo (OOT 2025), diagnóstico overfitting/GroupNorm, patch 256 |
| Avaliação | `eval_overnight.py`, `eval_combinado_sl.py` | Avaliação do acumulado vs referência nova e HLS; combinado Sentinel+Landsat |
| Figuras/tabelas/produtos | `figura_poligonos.py`, `png_cenarios.py`, `mapa_2025.py`, `tab_produtos.py`, `mensal_eficiencia.py`, `wiw_fraction.py`, `exportar_patches_qgis.py` | Saídas para o artigo e para o QGIS |

**Logs de execução (16, na raiz):** `OVERNIGHT_resultados.log`, `TIER_resultados.log`, `TVH_resultados.log`,
`DFORA_resultados.log`, `PERTILE_resultados.log`, `PAIRS_resultados.log`, `ACC005/ACC753/ACC_C20_resultados.log`,
`FINAL/KWB/NEWREF/PRED2025/SOPIX_resultados.log` + `README.md`.

> ⚠️ **Pasta `dados ` (com espaço no fim):** parece scratch de um `wget` interrompido (contém um
> arquivo `download?arq=...` de 411 M, um `.zip` de 0 byte, `wget-log` e uma duplicata de
> `referencia_hls.tif`). **Mas** também contém `limites_pantanal.gpkg`, que pode estar aberto no QGIS.
> **Não** mexer sem confirmação. Candidata nº 1 a limpeza — mediante aprovação explícita.

---

## Como os caminhos são montados hoje (entender ANTES de qualquer mudança)

Toda a configuração de caminhos vive em **um só lugar**, `geo_data_science.py` (linhas 35–65, 266–275):

```
RAIZ     = "/home/jovyan/ID-QUEIMADAS/jupyter"      # pai de guato_ba
DADOS    = f"{RAIZ}/bdc_outputs/2024/dados"          # ⚠️ FORA de guato_ba
GUATO    = f"{RAIZ}/guato_ba"                         # raiz do projeto
DATA_DIR = f"{GUATO}/data"                            # caches .npy (DENTRO)
HLS_MES  = f"{GUATO}/dados/referencia_hls_mes.tif"   # (DENTRO)
ACU_DIR  = f"{G.GUATO}/results/acumulado"             # (accumulate_tile.py:31)
```

Consequências:

1. **A maioria dos scripts usa `g.GUATO`/`G.GUATO`/`A.G.GUATO`** e monta caminhos como
   `f"{GUATO}/results/…"`, `f"{GUATO}/deeplearning/{modelo}/…"`, `f"{GUATO}/data/…"`.
   → **Renomear qualquer subpasta** (`deeplearning`→`models`, etc.) **quebra todas essas f-strings**,
   porque o nome da subpasta está escrito "na unha" depois de `{GUATO}`.
2. **Entradas externas FORA de `guato_ba`** (NÃO mover, NÃO versionar — são insumos do estudo):
   `bdc_outputs/…` (cubos, máscaras de referência fixas `REF_2024_*`/`REF_2025`), `acumulado_anual_2024/eventos_dini_dfim.tif` (HLS DOY),
   `pantanal_mgrs_v2.gpkg` (guardrail bioma), `DETER_NF_PANTANAL.gpkg`, `estudo_pantanal_2024/…` (MapBiomas).
   Coerente com a memória "referência fixa / não regerar".
3. **18 scripts** fazem `sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")` para achar o núcleo.
   → Mover o núcleo para `src/` obriga a trocar esses 18 inserts para `.../guato_ba/src`.
4. **Alguns scripts ignoram `GUATO` e usam caminho absoluto** (ver (c)) — esses precisam ser padronizados.

---

## (b) Tabela DE → PARA (itens principais)

Legenda de risco: 🟢 seguro (sem ref. em código, sem uso no QGIS) · 🟡 médio (ref. em código, reescrita mecânica) · 🔴 alto (muitas f-strings + provável uso no QGIS).

| DE (atual) | PARA (alvo) | Refs. em código | QGIS | Risco |
|---|---|---|---|---|
| `geo_data_science.py`, `redes.py`, `main.py`, `accumulate_tile.py`, `accumulate_landsat.py` | `src/` | 18 `sys.path.insert` + imports internos | não | 🟡 |
| `split_espacial*.py`, `sim_gsplit.py` | `scripts/split/` | só importam o núcleo | não | 🟡 |
| `acum2023.py`, `acum_gsplit.py`, `sweep2023.py`, `tile21kvt_2023.py` | `scripts/acumulado/` | importam núcleo | não | 🟡 |
| `ablacao_datasets.py`, `diag_5050.py`, `diag_gn.py`, `teste_256.py` | `scripts/ablacao/` | importam núcleo | não | 🟡 |
| `eval_overnight.py`, `eval_combinado_sl.py` | `scripts/eval/` | **ACU absoluto** | não | 🟡 |
| `figura_poligonos.py`, `png_cenarios.py`, `mapa_2025.py`, `tab_produtos.py`, `mensal_eficiencia.py`, `wiw_fraction.py`, `exportar_patches_qgis.py` | `scripts/figuras/` | f-strings + abs. | parte | 🟡 |
| `guato_ba_tutorial.ipynb` | `notebooks/` | — | não | 🟢 |
| `*_resultados.log` (16) | `logs/` | nenhum script lê | não | 🟢 |
| `reference/area_pantanal_0001.png` | mantém `reference/` | — | talvez | 🟢 |
| `artigo/` (tex, pdf, zips) | `docs/artigo/` (ou manter `artigo/`) | nenhum `.py` | não | 🟢 |
| `deeplearning/` | `models/deeplearning/` | **14+ f-strings `{GUATO}/deeplearning/…`** | sim | 🔴 |
| `machine_learning/` | `models/machine_learning/` | `mapa_2025.py:74` | sim | 🔴 |
| `hpo/` | `models/hpo/` (ou `hpo/`) | `ACU_DIR`/Optuna em disco | talvez | 🔴 |
| `data/` (caches .npy) | **manter `data/`** | `DATA_DIR` (geo_data_science) | não | 🔴 se renomear |
| `dados/` (entradas) | **manter `dados/`** | `HLS_MES`, `mensal_eficiencia` | **sim** | 🔴 se renomear |
| `results/` | **manter `results/`** | `ACU_DIR`, dezenas de f-strings | **sim** | 🔴 se renomear |
| `dados ` (espaço) ⚠️ | avaliar p/ remoção (scratch) | nenhum `.py` | `limites_pantanal.gpkg`? | 🟡 confirmar |
| `dissertação/` (vazia) | remover | — | não | 🟢 |
| `__pycache__/`, `.ipynb_checkpoints/` | remover + `.gitignore` | — | não | 🟢 |
| `artigo/Untitled Folder/` (vazia) | remover | — | não | 🟢 |

> **Recomendação central:** manter os nomes `data/`, `dados/`, `results/` **exatamente onde estão**
> (zero mudança de código, zero quebra de QGIS). O ganho de "git-ready" vem do `.gitignore` + agrupar
> scripts/logs + remover lixo — **não** de renomear as pastas pesadas de dados/saída.
> As linhas 🔴 só valem a pena via a estratégia de **config central + symlink** da seção (e).

### Estrutura-alvo (ideal)

```
guato_ba/                      # = GUATO (raiz do repo git)
├── src/                       # núcleo importável
│   ├── geo_data_science.py  redes.py  main.py
│   ├── accumulate_tile.py   accumulate_landsat.py
│   └── paths.py  (NOVO, proposto — fonte única de caminhos; ver (e))
├── scripts/                   # experimentos (split/ acumulado/ ablacao/ eval/ figuras/)
├── notebooks/                 # guato_ba_tutorial.ipynb, hpo_optuna_neptune.ipynb
├── data/        # caches .npy        (mantém nome; .gitignore)
├── dados/       # entradas rasters   (mantém nome; .gitignore)
├── models/      # ex-deeplearning/ + machine_learning/ + hpo/   (.pkl no .gitignore)
├── results/     # saídas             (mantém nome; rasters no .gitignore)
├── logs/        # *_resultados.log
├── reference/   docs/   README.md   .gitignore   requirements.txt(NOVO)
```

---

## (c) Scripts `.py` e linhas de caminho *hardcoded* que precisariam ser reescritas

> Serve para o usuário **dimensionar o risco** antes de aprovar. São as linhas que mudariam
> se a migração (mover núcleo p/ `src/` e/ou renomear pastas de dados) fosse executada.

### C1 — `sys.path.insert(0, ".../guato_ba")` → viraria `.../guato_ba/src` (18 ocorrências)

```
wiw_fraction.py:10        tile21kvt_2023.py:11     diag_5050.py:9
mensal_eficiencia.py:11   split_espacial_norte.py:9 acum_gsplit.py:11
figura_poligonos.py:12    eval_overnight.py:8      teste_256.py:8
eval_combinado_sl.py:8    diag_gn.py:10            sweep2023.py:11
ablacao_datasets.py:12    tab_produtos.py:9        acum2023.py:9
sim_gsplit.py:20          exportar_patches_qgis.py:11  split_espacial.py:14
```

### C2 — Configuração central em `geo_data_science.py` (ponto único a adaptar)

```
35: RAIZ       = "/home/jovyan/ID-QUEIMADAS/jupyter"
36: DADOS      = f"{RAIZ}/bdc_outputs/2024/dados"      # FORA de guato_ba
37: GUATO      = f"{RAIZ}/guato_ba"
38: BIOMA_GPKG = "/home/jovyan/ID-QUEIMADAS/pantanal_mgrs_v2.gpkg"
55-59: REF_2024_1/2/3, REF_2025  (máscaras fixas em bdc_outputs — FORA; NÃO regerar)
62: HLS_DOY  = f"{RAIZ}/acumulado_anual_2024/eventos_dini_dfim.tif"
63: HLS_MES  = f"{GUATO}/dados/referencia_hls_mes.tif"
64: DETER / 65: MAPB  (FORA de guato_ba)
266-269: CUBOS_2024 = {… f"{DADOS}/cubo_*_corrigido.tif"}   # FORA
273-274: MOS_2025 = {… f"{RAIZ}/bdc_outputs/2025/…"}        # FORA
275: DATA_DIR = f"{GUATO}/data"
```

### C3 — Caminhos ABSOLUTOS que ignoram `GUATO` (padronizar para config, independentemente da migração)

```
eval_combinado_sl.py:12    ACU = "/home/jovyan/.../guato_ba/results/acumulado"
eval_overnight.py:12       ACU = "/home/jovyan/.../guato_ba/results/acumulado"
png_cenarios.py:7          OUT = "/home/jovyan/.../guato_ba/results/artigo"
exportar_patches_qgis.py:18 OUT= "/home/jovyan/.../guato_ba/results/patches_qgis"
mensal_eficiencia.py:50    OUT = "/home/jovyan/.../guato_ba/dados/{tile.lower()}"
mensal_eficiencia.py:51    ckpt= "/home/jovyan/.../guato_ba/deeplearning/{modelo}/{modelo}_best.pkl"
tile21kvt_2023.py:107      HLS = "/home/jovyan/.../bdc_outputs/2023/aqm_2023.gpkg"   # entrada externa
acum2023.py:18             HLS2023 = "/home/jovyan/.../bdc_outputs/2023/aqm_2023.gpkg" # entrada externa
```

### C4 — f-strings `f"{GUATO}/<subpasta>/…"` que QUEBRAM se a subpasta for RENOMEADA

- **Se renomear `deeplearning/`** (→ `models/deeplearning/`): `mapa_2025.py:51`, `accumulate_landsat.py:153`,
  `accumulate_tile.py:148,191`, `ablacao_datasets.py:52,69`, `sim_gsplit.py:30`, `teste_256.py:20`,
  `diag_5050.py:15`, `diag_gn.py:17`, `acum2023.py:17`, `acum_gsplit.py:15`, `sweep2023.py:16`,
  `tile21kvt_2023.py:26`, `mensal_eficiencia.py:51` (absoluto).
- **Se renomear `results/`**: `accumulate_tile.py:31` (`ACU_DIR`), `main.py:337,423`, `split_espacial.py:22`,
  `split_espacial_norte.py:15`, `tile21kvt_2023.py:25`, `wiw_fraction.py:34`, `tab_produtos.py:16,17,59`,
  `sim_gsplit.py:89`, `figura_poligonos.py:26,28,29,35` + os absolutos de C3.
- **Se renomear `machine_learning/`**: `mapa_2025.py:74`.
- **Se renomear `data/`**: `geo_data_science.py:275` (`DATA_DIR`) + usos `317/321/335/341/349/354/355`.
- **Se renomear `dados/`**: `geo_data_science.py:63` (`HLS_MES`) + `mensal_eficiencia.py:50` (absoluto).

> **Leitura do risco:** mover o **núcleo** para `src/` = 18 linhas idênticas (mecânico, 🟡).
> Padronizar os **8 caminhos absolutos** de C3 = pontual (🟡).
> **Renomear pastas de dados/saída** (`deeplearning`→`models` etc.) = dezenas de f-strings **+ QGIS** (🔴) —
> só fazer via config central + symlink (seção (e)).

---

## (d) `.gitignore` sugerido

```gitignore
# ---- Python ----
__pycache__/
*.pyc
.ipynb_checkpoints/

# ---- Dados pesados / caches (regeneráveis ou insumos; NÃO versionar) ----
data/                 # caches .npy do dataset (15 G)
dados/                # rasters de entrada (4.8 G)
"dados "/             # scratch acidental (pasta com espaço) — confirmar remoção antes
*.npy
*.npz

# ---- Modelos (binários grandes) ----
deeplearning/**/*.pkl
machine_learning/**/*.pkl
hpo/_tmp_*/
*.pkl
# (manter versionados: historico*.csv, curva.png, hpo_best.json)

# ---- Rasters/vetores de saída (QGIS) ----
*.tif
*.tiff
*.tif.aux.xml
*.gpkg
*.gpkg-shm
*.gpkg-wal
*.nc
results/acumulado/_ckpt_*/   # caches .npz por tile

# ---- Artigo/LaTeX (manter só .tex e o .pdf final, se desejado) ----
artigo/*.aux
artigo/*.out
artigo/guato_ba_artigo.log
artigo/compile.log
artigo/*.zip          # pacotes Overleaf (grandes; opcional versionar)

# ---- Optuna (opcional: estudos reproduzíveis) ----
# hpo/*.db

# ---- Manter explicitamente (exceções úteis) ----
!results/artigo/*.csv   # tabelas do artigo (pequenas)
!**/historico*.csv
!**/hpo_best.json
!**/resumo_*.json
!logs/*.log
```

> Observação: como vários `.csv`/`.json`/`.png` **pequenos e significativos** convivem com rasters
> pesados, o `.gitignore` acima ignora por extensão pesada e **readmite** (`!…`) os leves que valem
> versionar. Vale um `git status` de teste (em branch) antes de confirmar.

---

## (e) Plano de migração em etapas seguras (PROPOSTA — requer aprovação)

**Princípios:** (i) usar `git mv` para preservar histórico; (ii) **não** renomear pasta pesada sem
symlink de compatibilidade; (iii) centralizar caminhos numa fonte única; (iv) validar QGIS a cada etapa.

**Etapa 0 — Inicializar git SEM mover nada.**
`git init` na raiz `guato_ba/`; adicionar o `.gitignore` de (d); `git add` só do leve; primeiro commit
"baseline". Isso já deixa "pronto para git" com risco zero.

**Etapa 1 — Fonte única de caminhos (`src/paths.py` ou reforço em `geo_data_science.py`).**
Introduzir constantes derivadas de uma variável de ambiente, p.ex.
`GUATO = os.environ.get("GUATO_ROOT", "/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")` e
`RESULTS_DIR`, `DEEP_DIR`, `ML_DIR`, `HPO_DIR`, `DATA_DIR`, `DADOS_DIR`. Trocar os **8 absolutos de C3**
e o `ACU_DIR` para usarem essas constantes. **Nada muda de lugar ainda** — só some o caminho "na unha".

**Etapa 2 — Núcleo → `src/` (🟡).**
`git mv` dos 5 módulos para `src/`; trocar os **18 `sys.path.insert`** (C1) para `.../guato_ba/src`
(ou instalar o pacote em modo editável e remover os inserts). Rodar `python -c "import geo_data_science"`
e um script de cada grupo para validar imports.

**Etapa 3 — Scripts/logs/notebooks → pastas (🟡/🟢).**
`git mv` dos 20 scripts para `scripts/<grupo>/`, dos 16 logs para `logs/`, do tutorial para `notebooks/`.
Sem mudança de caminho de dados — só de localização dos `.py`.

**Etapa 4 — (OPCIONAL, 🔴) Renomear pastas de dados/saída via symlink.**
Para `deeplearning`→`models/deeplearning` etc.: (a) `git mv`; (b) **criar symlink** do nome antigo apontando
para o novo (`ln -s models/deeplearning deeplearning`) para **QGIS e f-strings antigas continuarem achando**;
(c) só então reescrever as f-strings de C4 para as novas constantes da Etapa 1; (d) remover o symlink ao final,
depois de validar QGIS e rodar o pipeline ponta-a-ponta. **Se o ganho não compensar, pular a Etapa 4** e
manter os nomes atuais (recomendado).

**Etapa 5 — Limpeza de lixo (mediante confirmação, conforme memória de disco/QGIS).**
`__pycache__/`, `.ipynb_checkpoints/`, `dissertação/` (vazia), `artigo/Untitled Folder/` (vazia).
**`dados ` (com espaço): confirmar 1-a-1** — mover `limites_pantanal.gpkg` para `dados/` se estiver em uso no
QGIS; o resto (`download?arq=…`, zip de 0 byte, `wget-log`, duplicata de `referencia_hls.tif`) é candidato a
descarte **após o usuário confirmar**.

**Validação a cada etapa:** (1) `grep -rn "GUATO\|/home/jovyan"` deve bater com a config central;
(2) abrir o projeto no QGIS e conferir que nenhuma camada "quebrou"; (3) rodar `main.py` num tile e um script
de figura; (4) `git status` limpo (sem dado pesado rastreado).

---

## Pendências para alinhar com os outros agentes

> Espaços reservados para o orquestrador encaixar recomendações do **revisor de pipeline** e do
> **agente de neurocomputação**. Esta proposta de organização **não** decide esses pontos.

- **Onde guardar modelos NOVOS (neurocomputação):** definir se novos checkpoints vão para
  `models/deeplearning/<nome>/` (padrão atual) ou um esquema versionado (`models/<nome>/<data>/`).
  Confirmar convenção de nome `*_best.pkl`/`*_last.pkl` + `historico.csv` + `hpo_best.json`.
- **Scripts de melhoria/retreino (neurocomputação):** destino provável `scripts/treino/` ou `scripts/ablacao/`.
  Alinhar com os retreinos Tier2 em andamento (GroupNorm/SpecAug) antes de mover qualquer coisa — evitar
  quebrar jobs noturnos.
- **Pipeline (revisor):** validar se `src/` + import editável quebra algum fluxo (`accumulate_*`, `main.py`
  estágios); confirmar a lista final do `.gitignore` (especialmente `hpo/*.db` e `artigo/*.zip`).
- **Referências fixas:** reconfirmar que `bdc_outputs/…`, `pantanal_mgrs_v2.gpkg`, HLS/DETER/MapBiomas
  permanecem **fora** do repo e **nunca** são regerados (memória "referência fixa / não regerar").
- **Produto final combinado** (1.495.178 ha / 84,5%) e os rasters campeões (`biatt_unet`, THR 0.90, MMU 24):
  decidir com o revisor quais saídas de `results/acumulado/` são "oficiais" e merecem exceção de versionamento
  (ou um `results/FINAL/` curado), vs. as dezenas de variantes de ablação.

---
*Documento gerado como PROPOSTA. Nenhuma alteração foi feita no projeto além da criação deste arquivo.*
