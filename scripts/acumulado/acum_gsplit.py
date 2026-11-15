#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acumulado tile-a-tile do PANTANAL com a U-Net do SPLIT ESPACIAL honesto (unet_gsplit_norte),
para 2024, 2025 e 2026(jan→set). THR=0.50. GroupNorm (GUATO_NORM=gn setado ANTES de importar o modelo).
Saídas renomeadas para acum<ano>_unet_gsplit.{tif,gpkg} etc."""
import os, glob
os.environ["GUATO_NORM"]="gn"   # <<< GroupNorm ANTES de importar redes (via accumulate_tile)
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import accumulate_tile as A
from accumulate_tile import log, ACU_DIR

CKPT=f"{A.G.GUATO}/deeplearning/unet_gsplit_norte/unet_best.pkl"
assert os.path.exists(CKPT), CKPT
JOBS=[("2024","2024-01-01","2024-12-31"),
      ("2025","2025-01-01","2025-12-31"),
      ("2026","2026-01-01","2026-09-30")]

for yr,ini,fim in JOBS:
    A.INI=ini; A.FIM=fim
    tag=f"_gsplit{yr}"
    log(f"########## ACUMULADO {yr} ({ini}->{fim}) | unet_gsplit_norte THR=0.50 ##########")
    A.run_pantanal(modelo="unet", thr=0.50, ckpt=CKPT, tag=tag)
    # renomeia saídas acum2024_unet_gsplit<yr>.* -> acum<yr>_unet_gsplit.* (nome do ano correto)
    for f in glob.glob(f"{ACU_DIR}/*unet{tag}*"):
        b=os.path.basename(f)
        nb=b.replace("acum2024",f"acum{yr}").replace("prob2024",f"prob{yr}").replace("analisado_",f"analisado{yr}_").replace("detdoy_",f"detdoy{yr}_").replace(f"_gsplit{yr}","_gsplit")
        if nb!=b and not f.endswith(".npz") and "_ckpt_" not in f:
            os.rename(f, f"{ACU_DIR}/{nb}"); log(f"  renomeado -> {nb}")
    log(f"########## {yr} OK ##########")
log("===== ACUM_GSPLIT_ALL_OK =====")
