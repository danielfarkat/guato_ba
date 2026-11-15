#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SIMULAÇÃO split ESPACIAL honesto.
- Treino = SUL (~70% área) ; Validação = NORTE (~30% área) — via sidecar train_v123_256_SPLIT.npy.
- Receita campeã: unet (4 bandas) + GroupNorm + SpecAug + 128 pos/753 neg por batch.
- Seleção de época/early-stop/LR pelo IoU de VALIDAÇÃO (norte); val NUNCA entra no backprop.
- Calibra o THRESHOLD na validação do norte; aplica no OOT 2025.
- Teste final: 2025 (dentro do bioma, máscara curada).
"""
import os
# --- env ANTES de qualquer import de redes/main (GUATO_NORM é lido no import de redes) ---
os.environ["GUATO_NORM"]       = "gn"     # GroupNorm
os.environ["GUATO_SPECAUG"]    = "1"      # augmentação espectral
os.environ["GUATO_POS_PER_BATCH"] = "128"
os.environ["GUATO_NEG_PER_BATCH"] = "753"
os.environ["GUATO_SELECT"]     = "iou"    # seleção/early-stop/LR pelo IoU de val
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, torch
import geo_data_science as g
from geo_data_science import carregar_npy, ESC, IGN, metricas_pixel
import main as M
from redes import construir_dl, MODELOS_DL, _NORM

SPLIT="train_v123_256"; NOME="unet"
MASK=f"{g.DATA_DIR}/{SPLIT}_SPLIT.npy"
os.environ["GUATO_SPLITMASK"]=MASK
OUT=f"{g.GUATO}/deeplearning/unet_gsplit_norte"
g.log(f"SIM GSPLIT | modelo={NOME} norm={_NORM} | split={SPLIT} mask={os.path.basename(MASK)} | SELECT=iou | 128pos/753neg +SpecAug")
assert os.path.exists(MASK), MASK

# ---------- TREINO (sul) + VALIDAÇÃO (norte) ----------
best, bep, hist = M.treino_dl(NOME, {}, 150, OUT, tag="final", min_stop=40, patience=30,
                              split=SPLIT, live=True, log_every=2)
g.log(f"treino concluído: best(IoU val)={best:.4f} @ ép {bep}")

# ---------- carrega melhor modelo ----------
ck=torch.load(f"{OUT}/{NOME}_best.pkl", map_location="cuda", weights_only=False)
bandas=ck["bandas"]; mean=ck["mean"][None,:,None,None]; std=ck["std"][None,:,None,None]
model=construir_dl(NOME, p=0.0, pretrained=False).cuda(); model.load_state_dict(ck["state_dict"]); model.eval()

def preditor(T1,T2,Y):
    """retorna prob (sigmoid) e máscara de validade, em CPU float32/bool, por lote."""
    X=(np.concatenate([T1[:,bandas],T2[:,bandas]],1).astype(np.float32)/ESC-mean)/std
    P=np.zeros((len(X),X.shape[2],X.shape[3]),np.float32)
    with torch.no_grad():
        for i in range(0,len(X),16):
            xb=torch.from_numpy(X[i:i+16]).cuda(); yb=torch.from_numpy(Y[i:i+16].astype(np.int16)).cuda()
            xb=xb*(yb!=IGN).unsqueeze(1).float()
            with torch.cuda.amp.autocast(): pr=torch.sigmoid(model(xb)).float()
            P[i:i+16]=pr.cpu().numpy()
    return P

def met_thr(P,Y,thr):
    pr=(P>=thr); gt=(Y==1); vd=(Y!=IGN)
    TP=int((pr&gt).sum()); FP=int((pr&~gt&vd).sum()); FN=int((~pr&gt).sum()); TN=int((~pr&(Y==0)).sum())
    return metricas_pixel(TP,FP,FN,TN)

# ---------- calibra THRESHOLD na VALIDAÇÃO do norte ----------
isval=np.load(MASK).astype(bool)
T1,T2,Y,_=carregar_npy(SPLIT)
Pv=preditor(T1[isval],T2[isval],Y[isval]); Yv=Y[isval]
GRID=np.round(np.arange(0.05,0.96,0.05),2)
cal=[(t,met_thr(Pv,Yv,t)["IoU"]) for t in GRID]
thr_star=max(cal,key=lambda x:x[1])[0]
g.log("CALIB THR (val norte): "+" ".join(f"{t:.2f}:{iou:.3f}" for t,iou in cal))
mv=met_thr(Pv,Yv,thr_star); mv05=met_thr(Pv,Yv,0.5)
g.log(f"VAL norte  THR=0.50: IoU={mv05['IoU']:.4f} OE={mv05['OE']:.3f} CE={mv05['CE']:.3f}")
g.log(f"VAL norte  THR*={thr_star:.2f}: IoU={mv['IoU']:.4f} OE={mv['OE']:.3f} CE={mv['CE']:.3f}")
del T1,T2,Y,Pv

# ---------- TESTE OOT 2025 ----------
t1,t2,y,_=carregar_npy("test2025")
Pt=preditor(t1,t2,y)
for tag,t in [("THR=0.50",0.5),(f"THR*={thr_star:.2f}",thr_star)]:
    mm=met_thr(Pt,y,t)
    g.log(f"OOT2025 {tag}: IoU={mm['IoU']:.4f} OE={mm['OE']:.3f} CE={mm['CE']:.3f} rec={mm['recall']:.3f} prec={mm['precisao']:.3f}")

# resumo p/ csv
import pandas as pd
mmt05=met_thr(Pt,y,0.5); mmts=met_thr(Pt,y,thr_star)
pd.DataFrame([
    dict(cenario="val_norte", thr=0.50, IoU=mv05["IoU"], OE=mv05["OE"], CE=mv05["CE"]),
    dict(cenario="val_norte", thr=thr_star, IoU=mv["IoU"], OE=mv["OE"], CE=mv["CE"]),
    dict(cenario="oot2025",   thr=0.50, IoU=mmt05["IoU"], OE=mmt05["OE"], CE=mmt05["CE"]),
    dict(cenario="oot2025",   thr=thr_star, IoU=mmts["IoU"], OE=mmts["OE"], CE=mmts["CE"]),
]).to_csv(f"{g.GUATO}/results/split_espacial/resultado_gsplit_norte.csv", index=False)
g.log(f"resumo -> results/split_espacial/resultado_gsplit_norte.csv")
print("SIM_GSPLIT_OK")
