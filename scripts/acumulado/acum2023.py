#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acumulado tile-a-tile 2023 (unet_gsplit_norte, THR=0.50) + comparação vs HLS 2023 (aqm_2023.gpkg)."""
import os, glob
os.environ["GUATO_NORM"]="gn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import rasterize
import accumulate_tile as A
from accumulate_tile import log, ACU_DIR
import geo_data_science as g
from geo_data_science import grid_5880, mascara_bioma, PX_HA, metricas_artigo, DST

CKPT=f"{g.GUATO}/deeplearning/unet_gsplit_norte/unet_best.pkl"
HLS2023="/home/jovyan/ID-QUEIMADAS/jupyter/bdc_outputs/2023/aqm_2023.gpkg"

# ---- 1) acumulado 2023 ----
A.INI="2023-01-01"; A.FIM="2023-12-31"
log("########## ACUMULADO 2023 (2023-01-01->2023-12-31) | unet_gsplit_norte THR=0.50 ##########")
A.run_pantanal(modelo="unet", thr=0.50, ckpt=CKPT, tag="_gsplit2023")
for f in glob.glob(f"{ACU_DIR}/*unet_gsplit2023*"):
    b=os.path.basename(f)
    if "_ckpt_" in f or f.endswith(".npz"): continue
    nb=b.replace("acum2024","acum2023").replace("prob2024","prob2023").replace("analisado_","analisado2023_").replace("detdoy_","detdoy2023_").replace("_gsplit2023","_gsplit")
    if nb!=b: os.rename(f,f"{ACU_DIR}/{nb}"); log(f"  renomeado -> {nb}")
log("########## 2023 OK ##########")

# ---- 2) comparação vs HLS 2023 ----
GH,GW,GGT=grid_5880(); B=mascara_bioma()
def rd(f):
    with rasterio.open(f) as s: return s.read(1)>0
MOD=rd(f"{ACU_DIR}/acum2023_unet_gsplit.tif")&B
ANAL=rd(f"{ACU_DIR}/analisado2023_unet_gsplit.tif")&B
hg=gpd.read_file(HLS2023).to_crs(DST)
HLS=rasterize([(gm,1) for gm in hg.geometry if gm and not gm.is_empty],out_shape=(GH,GW),transform=GGT,fill=0,dtype="uint8").astype(bool)&B
log(f"[2023] modelo={MOD.sum()*PX_HA:,.0f}ha | analisado {ANAL.sum()/B.sum()*100:.1f}% | HLS2023={HLS.sum()*PX_HA:,.0f}ha")
for dn,dom in [("pantanal_inteiro",B),("analisado",ANAL)]:
    m=metricas_artigo(MOD,HLS,dom)
    log(f"[2023] vs HLS [{dn:16s}] IoU={m['IoU']:.3f} F1={m['F1']:.3f} OE={m['OE']:.3f} CE={m['CE']:.3f} area_mod={m['area_modelo_ha']:,.0f} area_ref={m['area_ref_ha']:,.0f}")
log("===== ACUM2023_OK =====")
