#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split espacial INVERTIDO: NORTE ~30% (área do bioma) = VALIDAÇÃO ; SUL ~70% = TREINO.
Reusa POS.npy (posições já reconstruídas e validadas) e Y.npy (p/ fração queimada) —
não relê cubos. Regera GPKG + PNG + sidecar _SPLIT.npy."""
import os
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, geopandas as gpd
from shapely.geometry import box, LineString
import geo_data_science as g
from geo_data_science import grid_5880, mascara_bioma, PATCH, BLOCK, IGN, RES, DST

SPLIT_NAME="train_v123_256"; OUTDIR=f"{g.GUATO}/results/split_espacial"; os.makedirs(OUTDIR,exist_ok=True)
GH,GW,GGT=grid_5880(); GX0,GYt=GGT.c,GGT.f; biom=mascara_bioma()
POS=np.load(f"{g.DATA_DIR}/{SPLIT_NAME}_POS.npy", allow_pickle=True)
Y=np.load(f"{g.DATA_DIR}/{SPLIT_NAME}_Y.npy")
g.log(f"SPLIT NORTE=VAL | {len(POS)} patches | reuso POS/Y (sem reler cubos)")

# fração queimada por patch (entre pixels válidos)
valid=(Y!=IGN); burn=(Y==1)
nv=valid.reshape(len(Y),-1).sum(1); nb=burn.reshape(len(Y),-1).sum(1)
BURN=(nb/np.maximum(1,nv)).astype(np.float32)

# limiar: 30% da ÁREA do bioma mais ao NORTE (acumula de cima p/ baixo)
rows=biom.sum(axis=1).astype(np.int64); tot=rows.sum()
prefix=np.cumsum(rows)                      # prefix[r]=área de row0(norte) até r
row_th=int(np.where(prefix>=0.30*tot)[0][0])   # 1a linha onde área-ao-norte >=30%
y_th=GYt-row_th*RES
g.log(f"corte 30% área NORTE: row={row_th} Y={y_th:.0f} | norte={prefix[row_th]/tot*100:.1f}% da área")

r0s=np.array([p[1] for p in POS],np.int64)
blk_center_row=(r0s//BLOCK+0.5)*BLOCK
is_val=blk_center_row < row_th              # bloco cujo centro está ao NORTE da linha -> VAL
n_val=int(is_val.sum()); n_tr=len(is_val)-n_val; pos=(BURN>0)
g.log(f"TREINO(sul)={n_tr} ({n_tr/len(is_val)*100:.1f}%)  VAL(norte)={n_val} ({n_val/len(is_val)*100:.1f}%)")
g.log(f"  patches c/ queimada: treino(sul)={int((pos&~is_val).sum())}  val(norte)={int((pos&is_val).sum())}")

np.save(f"{g.DATA_DIR}/{SPLIT_NAME}_SPLIT.npy", is_val.astype(np.uint8))
g.log("sidecar _SPLIT.npy atualizado (1=val=norte)")

geoms,recs=[],[]
for i,(par,r0,c0) in enumerate(POS):
    xmn=GX0+c0*RES; xmx=GX0+(c0+PATCH)*RES; ymx=GYt-r0*RES; ymn=GYt-(r0+PATCH)*RES
    geoms.append(box(xmn,ymn,xmx,ymx))
    recs.append(dict(split=("val" if is_val[i] else "treino"),par=f"datas_{par}",
                     burn_frac=round(float(BURN[i]),4),queimado=int(BURN[i]>0),
                     r0=int(r0),c0=int(c0)))
gdf=gpd.GeoDataFrame(recs,geometry=geoms,crs=DST)
gpkg=f"{OUTDIR}/patches_split_{SPLIT_NAME}.gpkg"
gdf.to_file(gpkg,layer="patches",driver="GPKG")
gpd.GeoDataFrame([{"tipo":"corte_30pct_norte"}],geometry=[LineString([(GX0,y_th),(GX0+GW*RES,y_th)])],crs=DST).to_file(gpkg,layer="linha_corte",driver="GPKG")
_,bgeom=g._bioma_geom()
gpd.GeoDataFrame([{"tipo":"bioma"}],geometry=[bgeom],crs=DST).to_file(gpkg,layer="bioma",driver="GPKG")
g.log(f"GPKG -> {gpkg}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,11))
gpd.GeoSeries([bgeom],crs=DST).boundary.plot(ax=ax,color="0.4",lw=0.8)
gdf[gdf.split=="treino"].plot(ax=ax,color="#2166AC",alpha=0.55,lw=0,label=f"treino (sul) n={n_tr}")
gdf[gdf.split=="val"].plot(ax=ax,color="#B2182B",alpha=0.55,lw=0,label=f"validação (norte) n={n_val}")
ax.plot([GX0,GX0+GW*RES],[y_th,y_th],color="k",ls="--",lw=1.2,label="corte 30% área (norte)")
ax.set_title(f"Split espacial 2024 — norte=validação / sul=treino\n{len(gdf)} patches 256px (v1+v2+v3)",fontsize=11)
ax.set_xlabel("E (m, EPSG:5880)"); ax.set_ylabel("N (m)"); ax.set_aspect("equal"); ax.legend(loc="upper right",fontsize=8)
plt.tight_layout(); png=f"{OUTDIR}/preview_split_{SPLIT_NAME}.png"; plt.savefig(png,dpi=130); plt.close()
g.log(f"PNG -> {png}"); print("SPLIT_NORTE_OK")
