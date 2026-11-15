#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablação dataset × modelo, julgada SOMENTE pelo TESTE OOT 2025 (patch-level).
Amostragem: batch 50/50 (queimada oversampled, distinta no batch + reusada entre batches;
não-queimada sem reposição) — ver treino_dl em main.py. LR cosine simples (sem warm-restarts).
Modelos: unet_nored (sem red) e unet (com red). Datasets: _v3 (par 0803->0902), _v12 (1,2), _v123 (1,2,3).
Cada célula: treina -> avalia no OOT 2025 (IoU/OE/CE/recall/precisão da classe queimada). Resumível.
Métrica de decisão = IoU da classe queimada no OOT 2025. Guardrail: ∩ bioma (embutido no dataset)."""
import os, time, json
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, torch
import geo_data_science as g
from geo_data_science import construir_dataset_2024, carregar_npy, PARES_2024, ESC, IGN, metricas_pixel
import main as M
from redes import construir_dl
GUATO=g.GUATO; DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
P1,P2,P3 = PARES_2024[0], PARES_2024[1], PARES_2024[2]
DATASETS = [("_v123aug",[P1,P2,P3],"train_v123")]   # teste A/B da augmentation radiométrica (reusa train_v123)
MODELOS = ["unet_nored"]                             # só o melhor (sem red); compara com baseline 0,646 (sem aug radiométrica)
HP={}; EPOCHS=1000; MINSTOP=100; PAT=40             # 1000 ép + early-stop pelo val_loss
_T=None   # cache do teste 2025 (T1,T2,Y)

def ts(): return time.strftime("%H:%M:%S")

def _load_test():
    global _T
    if _T is None:
        T1,T2,Y,_=carregar_npy("test2025"); _T=(T1,T2,Y)
        g.log(f"test2025: {len(Y)} patches | {int((Y==1).any(axis=(1,2)).sum())} com queimada")
    return _T

def oot2025(modelo, ckpt):
    T1,T2,Y=_load_test()
    ck=torch.load(ckpt, map_location=DEV, weights_only=False); bandas=ck["bandas"]
    mean=ck["mean"][None,:,None,None]; std=ck["std"][None,:,None,None]
    m=construir_dl(modelo,p=0.0,pretrained=False).to(DEV); m.load_state_dict(ck["state_dict"]); m.eval()
    X=(np.concatenate([T1[:,bandas],T2[:,bandas]],axis=1).astype(np.float32)/ESC - mean)/std
    TP=FP=FN=TN=0
    with torch.no_grad():
        for i in range(0,len(X),64):
            xb=torch.from_numpy(X[i:i+64]).to(DEV); yb=torch.from_numpy(Y[i:i+64].astype(np.int16)).to(DEV)
            xb=xb*(yb!=IGN).unsqueeze(1).float()
            with torch.cuda.amp.autocast(): pr=(torch.sigmoid(m(xb))>=0.5)
            vd=(yb!=IGN); gt=(yb==1)
            TP+=int((pr&gt).sum());FP+=int((pr&~gt&vd).sum());FN+=int((~pr&gt).sum());TN+=int((~pr&(yb==0)).sum())
    del m; torch.cuda.empty_cache()
    return metricas_pixel(TP,FP,FN,TN)

def treina_e_avalia(modelo, vtag, pares, pfx):
    outdir=f"{GUATO}/deeplearning/{modelo}{vtag}"
    done=f"{outdir}/{modelo}_last.pkl"; ckpt=f"{outdir}/{modelo}_best.pkl"; resj=f"{outdir}/oot2025.json"
    g.log(f"===== CELULA {modelo}{vtag} (pares={[p[0] for p in pares]}) {ts()} =====")
    if os.path.exists(resj):
        g.log(f"[{modelo}{vtag}] já avaliado — pulando"); return
    if not os.path.exists(done):
        os.makedirs(outdir, exist_ok=True)
        construir_dataset_2024(forcar=False, pares=pares, prefixo=pfx)
        best,bep,_ = M.treino_dl(modelo, HP, EPOCHS, outdir, tag="final",
                                 min_stop=MINSTOP, patience=PAT, split=pfx, live=True)
        g.log(f"[{modelo}{vtag}] treino concluído: best val_loss={best:.4f}@{bep}")
    mm=oot2025(modelo, ckpt); json.dump(mm, open(resj,"w"), indent=2)
    g.log(f"[{modelo}{vtag}] OOT2025: IoU={mm['IoU']:.4f} F1={mm['F1']:.4f} OE={mm['OE']:.3f} CE={mm['CE']:.3f} rec={mm['recall']:.3f} prec={mm['precisao']:.3f}")

def tabela(cells):
    rows=[]
    for modelo,vtag in cells:
        resj=f"{GUATO}/deeplearning/{modelo}{vtag}/oot2025.json"
        if not os.path.exists(resj): print(f"  (falta {modelo}{vtag})"); continue
        r=json.load(open(resj)); rows.append((f"{modelo}{vtag}", r))
    print(f"\n{'CELULA':20} | {'IoU_2025':>8} {'F1':>6} {'OE':>6} {'CE':>6} {'recall':>7} {'prec':>6}")
    for nome,r in rows:
        print(f"{nome:20} | {r['IoU']:>8.3f} {r['F1']:>6.3f} {r['OE']:>6.3f} {r['CE']:>6.3f} {r['recall']:>7.3f} {r['precisao']:>6.3f}")
    if rows:
        b=max(rows,key=lambda x:x[1]['IoU'])
        print(f"\n>>> MELHOR no OOT 2025: {b[0]} (IoU={b[1]['IoU']:.3f}, OE={b[1]['OE']:.3f}, CE={b[1]['CE']:.3f})")
    print("--- referência (técnicas anteriores, unet_nored_v3 OOT2025): 10/90 IoU=0.581 | 70/30+batch280 IoU=0.615 ---")

if __name__=="__main__":
    cells=[(m,v[0]) for m in MODELOS for v in DATASETS]
    for modelo in MODELOS:
        for vtag,pares,pfx in DATASETS:
            treina_e_avalia(modelo,vtag,pares,pfx)
    tabela(cells)
    print("=== ABLACAO COMPLETA ===")
