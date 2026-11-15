#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exporta os PATCHES de treino georreferenciados p/ auditar no QGIS (por par de datas).
Mesmo tiling do dataset (step=128, SCL_MIN=0.30, válido = SCL+WIW nas 2 datas ∩ bioma).
Saída em results/patches_qgis/:
  patches_label_datas<N>.tif  -> 1=queimada(treinada), 2=não-queimada, 3=ignorado(nuvem/água), 0=fora de patch
  patches_footprint_datas<N>.gpkg -> caixas 128x128 dos patches (atrib: grupo, queimada 0/1, frac_valido)"""
import os
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, rasterio, geopandas as gpd
from rasterio.windows import bounds as win_bounds, Window
from shapely.geometry import box
import geo_data_science as g
from geo_data_science import (grid_5880, mascara_bioma, keep_valido, _ler_cubo, _rasteriza_ref,
                              salvar_raster, CUBOS_2024, PARES_2024, SCL_MIN, PATCH, BLOCK, DST, PX_HA)
OUT="/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba/results/patches_qgis"; os.makedirs(OUT, exist_ok=True)

def exporta_par(nome, k1, k2, ref_arq):
    GH,GW,GGT=grid_5880(); biom=mascara_bioma()
    g.log(f"[datas_{nome}] lendo cubos {k1}->{k2}…")
    d1,GT,H,W=_ler_cubo(CUBOS_2024[k1]); d2,_,_,_=_ler_cubo(CUBOS_2024[k2])
    val = keep_valido(d1["b8a"],d1["b11"],d1["b12"],d1["scl"]) & keep_valido(d2["b8a"],d2["b11"],d2["b12"],d2["scl"])
    valb = val & biom
    ref = _rasteriza_ref(ref_arq, GGT, GH, GW)
    out = np.zeros((GH,GW), np.uint8); geoms=[]; grp=[]; qm=[]; fv=[]
    npatch=0; nqueima=0
    for r0 in range(0, H-PATCH+1, PATCH):
        for c0 in range(0, W-PATCH+1, PATCH):
            sl=(slice(r0,r0+PATCH), slice(c0,c0+PATCH))
            vp=valb[sl]; frac=float(vp.mean())
            if frac < SCL_MIN: continue
            rp=ref[sl]
            lab=np.where(vp & rp, 1, np.where(vp, 2, 3)).astype(np.uint8)
            out[sl]=lab
            hb=int((rp & vp).any())
            b=win_bounds(Window(c0,r0,PATCH,PATCH), GGT)
            geoms.append(box(*b)); grp.append(f"{r0//BLOCK}_{c0//BLOCK}"); qm.append(hb); fv.append(round(frac,3))
            npatch+=1; nqueima+=hb
    salvar_raster(f"{OUT}/patches_label_datas{nome}.tif", out, nodata=0, dtype="uint8")
    gdf=gpd.GeoDataFrame({"grupo":grp,"queimada":qm,"frac_valido":fv}, geometry=geoms, crs=DST)
    gdf.to_file(f"{OUT}/patches_footprint_datas{nome}.gpkg", driver="GPKG")
    g.log(f"[datas_{nome}] ✅ {npatch} patches ({nqueima} com queimada) -> {OUT}")
    del d1,d2,val,valb,ref,out

if __name__=="__main__":
    alvo = sys.argv[1] if len(sys.argv)>1 else None   # ex: "3" p/ só datas_3; vazio = todos
    for nome,k1,k2,ref_arq in PARES_2024:
        if alvo and nome!=alvo: continue
        exporta_par(nome,k1,k2,ref_arq)
    print("=== PATCHES QGIS OK ===")
