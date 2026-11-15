#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preenche a tabela 'Acumulado 2024 do modelo x produtos, tile a tile' (HLS/DETER/MapBiomas).
Colunas: Correlacao (Pearson das áreas queimadas POR TILE), IoU (pixel, classe queimada), Area_ha (produto).
Tudo dentro do bioma. Modelo = acum2024_unet.tif (a U-Net da tabela do artigo)."""
import os
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, rasterio, geopandas as gpd, pandas as pd
from rasterio.features import rasterize
import geo_data_science as g
from geo_data_science import grid_5880, mascara_bioma, PX_HA, DST, carregar_deter, carregar_mapbiomas, carregar_hls_mes

GH,GW,GGT = grid_5880(); BIOMA = mascara_bioma()
MODEL_TIF = f"{g.GUATO}/results/acumulado/acum2024_unet.tif"
ANAL_TIF  = f"{g.GUATO}/results/acumulado/analisado_unet.tif"
with rasterio.open(ANAL_TIF) as s: ANAL = s.read(1) > 0
B = ANAL & BIOMA   # ÁREA ÚTIL = domínio analisado (SCL válido) ∩ bioma — NÃO o bioma inteiro
g.log(f"TAB PRODUTOS | modelo={os.path.basename(MODEL_TIF)} | DOMÍNIO=área útil (analisado∩bioma)={int(B.sum())*PX_HA:,.0f} ha de {int(BIOMA.sum())*PX_HA:,.0f} ha ({B.sum()/BIOMA.sum()*100:.1f}%)")

with rasterio.open(MODEL_TIF) as s:
    md = s.read(1)
    if md.shape != (GH,GW):   # alinhar ao grid se necessário
        from rasterio.warp import reproject, Resampling
        o=np.zeros((GH,GW), md.dtype); reproject(rasterio.band(s,1),o,src_transform=s.transform,src_crs=s.crs,dst_transform=GGT,dst_crs=DST,resampling=Resampling.nearest); md=o
model = (md > 0) & B
g.log(f"modelo: {int(model.sum())*PX_HA:,.0f} ha")

prods = {}
g.log("carregando HLS…");       prods["HLS"]       = (carregar_hls_mes() > 0) & B
g.log("carregando DETER…");     prods["DETER"]     = carregar_deter(2024) & B
g.log("carregando MapBiomas…"); prods["MapBiomas Fogo"] = carregar_mapbiomas() & B

# tiles MGRS p/ correlação por tile
pg,_ = g._bioma_geom()
g.log(f"tiles: {len(pg)}")
tile_masks = []
for geom in pg.geometry.values:
    tm = rasterize([(geom,1)], out_shape=(GH,GW), transform=GGT, fill=0, dtype="uint8").astype(bool) & B
    tile_masks.append(tm)

def iou(a,b):
    inter=int((a&b).sum()); union=int((a|b).sum()); return inter/union if union else 0.0
def corr_por_tile(a,b):
    xa=np.array([ (a&tm).sum()*PX_HA for tm in tile_masks ], float)
    xb=np.array([ (b&tm).sum()*PX_HA for tm in tile_masks ], float)
    keep = (xa+xb) > 0           # ignora tiles sem queimada em ambos
    if keep.sum() < 3: return float("nan"), int(keep.sum())
    r = float(np.corrcoef(xa[keep], xb[keep])[0,1]); return r, int(keep.sum())

rows=[]
for nome,p in prods.items():
    r,nt = corr_por_tile(model,p); iu=iou(model,p); ar=int(p.sum())*PX_HA
    g.log(f"  Modelo x {nome}: corr(tile)={r:.3f} (n={nt}) IoU={iu:.3f} area={ar:,.0f} ha")
    rows.append(dict(comparacao=f"Modelo x {nome}", correlacao=round(r,3), IoU=round(iu,3),
                     area_ha=round(ar,2), area_modelo_ha=round(int(model.sum())*PX_HA,2), n_tiles=nt))

out=f"{g.GUATO}/results/artigo/tab_produtos.csv"
pd.DataFrame(rows).to_csv(out, index=False)
g.log(f"CSV -> {out}")
print(pd.DataFrame(rows).to_string(index=False))
print("TAB_PRODUTOS_OK")
