#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split ESPACIAL norte/sul do dataset 2024 (train_v123_256).
- Norte ~70% = TREINO ; Sul ~30% (por ÁREA do bioma) = VALIDAÇÃO (holdout espacial).
- Reconstrói as posições EXATAS de cada patch relendo os cubos na MESMA ordem de
  construir_dataset_2024 (pares v1,v2,v3, step=PATCH, SCL_MIN), valida contra o G salvo.
- Atribui split por BLOCO (grupo espacial _br_bc, 2560px) p/ não vazar na fronteira.
- Gera GPKG (footprints dos patches + linha de corte + contorno do bioma) e PNG de preview.
- Salva sidecar train_v123_256_POS.npy (par,r0,c0) e _SPLIT.npy (1=val).
"""
import os
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, geopandas as gpd, pandas as pd
from shapely.geometry import box, LineString
import geo_data_science as g
from geo_data_science import (grid_5880, mascara_bioma, keep_valido, _ler_cubo,
                              _rasteriza_ref, PARES_2024, CUBOS_2024, PATCH, SCL_MIN, BLOCK, IGN, PX_HA, RES, DST)

SPLIT_NAME = "train_v123_256"
OUTDIR = f"{g.GUATO}/results/split_espacial"; os.makedirs(OUTDIR, exist_ok=True)
g.log(f"SPLIT ESPACIAL | pool={SPLIT_NAME} | PATCH={PATCH} SCL_MIN={SCL_MIN} BLOCK={BLOCK}")

GH, GW, GGT = grid_5880(); GX0, GYt = GGT.c, GGT.f
biom = mascara_bioma()

# ---- 1) limiar 30% da ÁREA do bioma mais ao sul (linha de northing constante) ----
rows = biom.sum(axis=1).astype(np.int64)         # px de bioma por linha (row0=norte)
tot = rows.sum(); suffix = np.cumsum(rows[::-1])[::-1]   # suffix[r]=área (px) de r até o sul
row_th = int(np.where(suffix >= 0.30*tot)[0][-1])        # maior row c/ área-ao-sul >=30%
y_th = GYt - row_th*RES
g.log(f"corte 30% área sul: row={row_th} Y={y_th:.0f} | sul={suffix[row_th]/tot*100:.1f}% da área do bioma")

# ---- 2) reconstrução EXATA das posições (mesma ordem de construir_dataset_2024) ----
cache = {}
def ler(k):
    if k not in cache: cache[k] = _ler_cubo(CUBOS_2024[k])[0]
    return cache[k]

POS = []          # (par, r0, c0)
BURN = []         # fração queimada entre válidos do patch
GREC = []         # grupo bloco reconstruído "br_bc"
for nome, k1, k2, ref_arq in PARES_2024:
    g.log(f"  par datas_{nome} ({k1}->{k2})…")
    d1 = ler(k1); d2 = ler(k2)
    ref = _rasteriza_ref(ref_arq, GGT, GH, GW)
    val = keep_valido(d1["b8a"], d1["b11"], d1["b12"], d1["scl"]) & \
          keep_valido(d2["b8a"], d2["b11"], d2["b12"], d2["scl"])
    valb = val & biom
    H, W = d1["scl"].shape
    for r0 in range(0, H-PATCH+1, PATCH):
        for c0 in range(0, W-PATCH+1, PATCH):
            sl = (slice(r0, r0+PATCH), slice(c0, c0+PATCH))
            vp = valb[sl]
            if vp.mean() < SCL_MIN: continue
            nv = int(vp.sum()); nb = int((ref[sl] & vp).sum())
            POS.append((nome, r0, c0)); BURN.append(nb/max(1,nv))
            GREC.append(f"{r0//BLOCK}_{c0//BLOCK}")
    del ref, val, valb
POS = np.array(POS, dtype=object); BURN = np.array(BURN, np.float32); GREC = np.array(GREC)
g.log(f"reconstruídos {len(POS)} patches")

# ---- 3) validar contra o G salvo do dataset ----
_,_,Y,Gsav = g.carregar_npy(SPLIT_NAME)
assert len(Gsav) == len(POS), f"contagem difere: salvo={len(Gsav)} recon={len(POS)}"
ok = int((Gsav == GREC).sum())
g.log(f"validação G: {ok}/{len(Gsav)} blocos batem ({ok/len(Gsav)*100:.1f}%)")
assert ok == len(Gsav), "ordem/posição não bate com o dataset salvo!"

# ---- 4) split por BLOCO (grupo inteiro p/ um lado) pelo CENTRO do bloco ----
r0s = np.array([p[1] for p in POS], np.int64)
blk_row = r0s // BLOCK
blk_center_row = (blk_row + 0.5) * BLOCK
is_val = blk_center_row >= row_th      # bloco cujo centro está ao sul da linha -> VALIDAÇÃO
n_val = int(is_val.sum()); n_tr = len(is_val)-n_val
# positivos (patch com queimada) por lado
pos_mask = (BURN > 0)
g.log(f"TREINO(norte)={n_tr} ({n_tr/len(is_val)*100:.1f}%)  VAL(sul)={n_val} ({n_val/len(is_val)*100:.1f}%)")
g.log(f"  patches c/ queimada: treino={int((pos_mask&~is_val).sum())}  val={int((pos_mask&is_val).sum())}")

# ---- 5) sidecars p/ reuso no treino ----
np.save(f"{g.DATA_DIR}/{SPLIT_NAME}_POS.npy", POS)
np.save(f"{g.DATA_DIR}/{SPLIT_NAME}_SPLIT.npy", is_val.astype(np.uint8))
g.log(f"sidecars salvos: {SPLIT_NAME}_POS.npy, {SPLIT_NAME}_SPLIT.npy")

# ---- 6) GPKG: footprints dos patches ----
geoms, recs = [], []
for i,(par,r0,c0) in enumerate(POS):
    xmn = GX0 + c0*RES; xmx = GX0 + (c0+PATCH)*RES
    ymx = GYt - r0*RES; ymn = GYt - (r0+PATCH)*RES
    geoms.append(box(xmn,ymn,xmx,ymx))
    recs.append(dict(split=("val" if is_val[i] else "treino"), par=f"datas_{par}",
                     burn_frac=round(float(BURN[i]),4), queimado=int(BURN[i]>0),
                     blk=GREC[i], r0=int(r0), c0=int(c0)))
gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=DST)
gpkg = f"{OUTDIR}/patches_split_{SPLIT_NAME}.gpkg"
gdf.to_file(gpkg, layer="patches", driver="GPKG")
# linha de corte + contorno bioma como camadas de contexto
line = LineString([(GX0, y_th), (GX0+GW*RES, y_th)])
gpd.GeoDataFrame([{"tipo":"corte_30pct_sul"}], geometry=[line], crs=DST).to_file(gpkg, layer="linha_corte", driver="GPKG")
_, bgeom = g._bioma_geom()
gpd.GeoDataFrame([{"tipo":"bioma"}], geometry=[bgeom], crs=DST).to_file(gpkg, layer="bioma", driver="GPKG")
g.log(f"GPKG -> {gpkg} ({len(gdf)} patches, 3 camadas)")

# ---- 7) PNG preview ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,11))
gpd.GeoSeries([bgeom], crs=DST).boundary.plot(ax=ax, color="0.4", lw=0.8)
gdf[gdf.split=="treino"].plot(ax=ax, color="#2166AC", alpha=0.55, lw=0, label=f"treino (norte) n={n_tr}")
gdf[gdf.split=="val"].plot(ax=ax, color="#B2182B", alpha=0.55, lw=0, label=f"validação (sul) n={n_val}")
ax.plot([GX0, GX0+GW*RES],[y_th,y_th], color="k", ls="--", lw=1.2, label="corte 30% área (sul)")
ax.set_title(f"Split espacial 2024 — norte=treino / sul=validação\n{len(gdf)} patches 256px (v1+v2+v3)", fontsize=11)
ax.set_xlabel("E (m, EPSG:5880)"); ax.set_ylabel("N (m)"); ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); png=f"{OUTDIR}/preview_split_{SPLIT_NAME}.png"; plt.savefig(png, dpi=130); plt.close()
g.log(f"PNG -> {png}")
print("SPLIT_ESPACIAL_OK")
