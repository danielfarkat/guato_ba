#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Teste de PATCH 256: reconstrói o dataset v123 e o teste 2025 em 256x256, treina unet_nored (sem red)
a fundo (1000 ép, early-stop no val_loss) e avalia no OOT 2025. Compara com baseline 128 (IoU 0.646)."""
import os, json, time
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, torch
import geo_data_science as g
from geo_data_science import construir_dataset_2024, construir_teste_2025, carregar_npy, PARES_2024, ESC, IGN, metricas_pixel, PATCH
import main as M
from redes import construir_dl
P=PARES_2024
g.log(f"PATCH atual = {PATCH} (deve ser 256)")
g.log("1) reconstruindo dataset v123 (256) + teste 2025 (256)...")
construir_dataset_2024(forcar=True, pares=[P[0],P[1],P[2]], prefixo="train_v123_256")
construir_teste_2025(forcar=True)
g.log("2) treinando unet_nored em 256 (1000 ép, early-stop val_loss)...")
outdir=f"{g.GUATO}/deeplearning/unet_nored_v123_256"
best,bep,_=M.treino_dl("unet_nored", {}, 1000, outdir, tag="final", min_stop=100, patience=40, split="train_v123_256", live=True)
g.log(f"treino 256 concluído: best val_loss={best:.4f}@{bep}")
g.log("3) avaliando OOT 2025 (256)...")
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
mm=metricas_pixel(TP,FP,FN,TN); json.dump(mm, open(f"{outdir}/oot2025.json","w"), indent=2)
print(f"OOT2025 patch256: IoU={mm['IoU']:.4f} F1={mm['F1']:.4f} OE={mm['OE']:.3f} CE={mm['CE']:.3f} rec={mm['recall']:.3f} prec={mm['precisao']:.3f} | baseline128 IoU=0.646 OE=0.346 CE=0.013")
print("=== TESTE 256 OK ===")
