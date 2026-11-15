#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sensibilidade à nuvem (2023): varia o filtro de nuvem de CENA (<=1,10,20,30%), área-limpa FIXA 95%.
Tiles representativos de muito fogo; tile a tile + união (run() já dissolve os tiles dados).
Reporta área queimada e cobertura por nível."""
import os
os.environ["GUATO_NORM"]="gn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import pandas as pd
import accumulate_tile as A
from accumulate_tile import log

CKPT=f"{A.G.GUATO}/deeplearning/unet_gsplit_norte/unet_best.pkl"
TILES=["21KVT","21KVU","21KVV","21KWB"]   # maior fogo
NIVEIS=[1,10,20,30]                        # % nuvem de cena (<=); 1 ~ "0" (clear)
A.INI="2023-01-01"; A.FIM="2023-12-31"
log(f"########## SWEEP NUVEM 2023 | tiles={TILES} | área-limpa FIXA 95% | níveis={NIVEIS}% ##########")
rows=[]
for cl in NIVEIS:
    log(f"===== nuvem <= {cl}% =====")
    r=A.run(modelo="unet", tiles=TILES, cloud=float(cl), in_bioma_min=0.95, thr=0.50, ckpt=CKPT, tag=f"_sw2023_c{cl}")
    rows.append(dict(nuvem_pct=cl, area_queimada_ha=round(r["area_queimada_ha"],0), cobertura_pct=round(r["cobertura_pantanal_pct"],2)))
    log(f"  -> nuvem<={cl}%: {r['area_queimada_ha']:,.0f} ha | cobertura {r['cobertura_pantanal_pct']:.2f}%")
df=pd.DataFrame(rows)
out=f"{A.G.GUATO}/results/artigo/sweep_nuvem_2023.csv"; df.to_csv(out,index=False)
log(f"CSV -> {out}")
print(df.to_string(index=False)); print("SWEEP2023_OK")
