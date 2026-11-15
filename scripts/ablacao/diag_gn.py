#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnóstico GroupNorm: 20 épocas FIXAS, batch 50/50, patch 256 (reusa train_v123_256), mas com
GroupNorm no lugar do BatchNorm (GUATO_NORM=gn). Objetivo: ver se o gap train-loss vs val-loss some/cai
(o BN inflava o val por usar running stats). Gera curva.png e avalia OOT 2025."""
import os
os.environ["GUATO_NORM"] = "gn"   # <<< ativa GroupNorm ANTES de importar redes
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import json, numpy as np, torch
import geo_data_science as g
from geo_data_science import carregar_npy, ESC, IGN, metricas_pixel
import main as M
from redes import construir_dl, _NORM
g.log(f"DIAG GroupNorm | _NORM={_NORM} | 20 ép FIXAS | 50/50 | patch 256")
outdir=f"{g.GUATO}/deeplearning/unet_nored_v123_256_gn"
best,bep,_=M.treino_dl("unet_nored", {}, 20, outdir, tag="final", min_stop=999, patience=999, split="train_v123_256", live=True, log_every=1)
g.log(f"treino GN (20ep): best val_loss={best:.4f}@{bep}")
ck=torch.load(f"{outdir}/unet_nored_best.pkl", map_location="cuda", weights_only=False); bandas=ck["bandas"]
mean=ck["mean"][None,:,None,None]; std=ck["std"][None,:,None,None]
m=construir_dl("unet_nored",p=0.0,pretrained=False).cuda(); m.load_state_dict(ck["state_dict"]); m.eval()
T1,T2,Y,_=carregar_npy("test2025")
X=(np.concatenate([T1[:,bandas],T2[:,bandas]],1).astype(np.float32)/ESC-mean)/std
TP=FP=FN=TN=0
with torch.no_grad():
    for i in range(0,len(X),16):
        xb=torch.from_numpy(X[i:i+16]).cuda(); yb=torch.from_numpy(Y[i:i+16].astype(np.int16)).cuda()
        xb=xb*(yb!=IGN).unsqueeze(1).float()
        with torch.cuda.amp.autocast(): pr=(torch.sigmoid(m(xb))>=0.5)
        vd=(yb!=IGN);gt=(yb==1)
        TP+=int((pr&gt).sum());FP+=int((pr&~gt&vd).sum());FN+=int((~pr&gt).sum());TN+=int((~pr&(yb==0)).sum())
mm=metricas_pixel(TP,FP,FN,TN)
print(f"OOT2025 (GroupNorm,50/50,256,20ep): IoU={mm['IoU']:.4f} OE={mm['OE']:.3f} CE={mm['CE']:.3f} rec={mm['recall']:.3f} prec={mm['precisao']:.3f}")
print("=== DIAG GN OK (ver curva.png: train vs val) ===")
