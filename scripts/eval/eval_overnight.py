#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Avaliação do acumulado 2024 Sentinel e Landsat vs REFERÊNCIA NOVA (2 pares) e vs HLS.
Métrica = IoU da CLASSE QUEIMADA (+ OE omissão, CE comissão), nos domínios Pantanal inteiro e analisado."""
import os
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, rasterio
import geo_data_science as g
from geo_data_science import grid_5880, mascara_bioma, carregar_referencia, carregar_hls_mes, PX_HA
ACU="/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba/results/acumulado"
GH,GW,GGT=grid_5880(); B=mascara_bioma()
REF=carregar_referencia(2024)&B
HLS=(carregar_hls_mes()>0)&B
def rd(f):
    if not os.path.exists(f): return None
    with rasterio.open(f) as s: return (s.read(1)>0)&B
def met(M,A,dom,nome):
    M=M&dom; A=A&dom
    a=float((M&A).sum()); b=float((M&~A).sum()); c=float((~M&A).sum())
    iou=a/(a+b+c) if (a+b+c) else 0; oe=c/(a+c) if (a+c) else 0; ce=b/(a+c) if (a+c) else 0
    print(f"    {nome:26s} IoU_queimada={iou:.3f}  OE={oe:.3f}  CE={ce:.3f} | TP={int(a):>10,}px FP={int(b):>10,}px FN={int(c):>10,}px")
def avaliar(tag, titulo):
    M=rd(f"{ACU}/acum2024_biatt_unet{tag}.tif"); AN=rd(f"{ACU}/analisado_biatt_unet{tag}.tif")
    if M is None: print(f"\n== {titulo}: (ausente {tag}) =="); return
    cob=AN.sum()/max(B.sum(),1)*100
    print(f"\n== {titulo} == area={M.sum()*PX_HA:,.0f}ha | cobertura={cob:.1f}%")
    print("  vs REFERENCIA (2 pares, 30/05->02/09):")
    met(M,REF,B,"Pantanal inteiro"); met(M,REF,AN,"dominio analisado")
    print("  vs HLS (ano todo):")
    met(M,HLS,B,"Pantanal inteiro"); met(M,HLS,AN,"dominio analisado")
print(f"REF(2 pares)={REF.sum()*PX_HA:,.0f}ha | HLS={HLS.sum()*PX_HA:,.0f}ha | BIOMA={B.sum()*PX_HA:,.0f}ha")
avaliar("_sent2024", "SENTINEL 2024 (biatt_unet, sem red)")
avaliar("_landsat_pantanal", "LANDSAT 2024 (biatt_unet transfer, sem red)")
print("\n=== EVAL OVERNIGHT COMPLETO ===")
