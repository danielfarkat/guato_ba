#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guato_ba · mapa_2025.py — máscara de queimada 2025 (mapa COMPLETO) por modelo, p/ análise no QGIS.

Inferência sobre o mosaico 2025 (par 04/09 -> 31/10), dentro do Pantanal, entrada sem água/nuvem.
DL: janela deslizante. ML: predição por pixel em chunks. Saída: GeoTIFF 5880 + overviews por modelo
em results/mascaras_2025/mascara2025_<modelo>.tif, recortado ao bioma, filtro >=12px.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, rasterio, torch
import geo_data_science as G
from geo_data_science import (log, ESC, IGN, PX_HA, DST, keep_valido, indices, refl,
                              _bioma_no_grid, filtrar_min_px, MOS_2025)
from rasterio.warp import Resampling
THR = 0.5; MIN_PX = 12; PATCH = 128; OV = 16; STP = PATCH - OV; BATCH = 64
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = f"{G.GUATO}/results/mascaras_2025"; os.makedirs(OUT, exist_ok=True)
BAND = G.BAND

def _carrega_2025():
    with rasterio.open(MOS_2025["t1"]) as s:
        GT = s.transform; H, W = s.shape
        d1 = {k: s.read(BAND[k]) for k in ("red", "b8a", "b11", "b12", "scl")}
    with rasterio.open(MOS_2025["t2"]) as s:
        d2 = {k: s.read(BAND[k]) for k in ("red", "b8a", "b11", "b12", "scl")}
    val = keep_valido(d1["b8a"], d1["b11"], d1["b12"], d1["scl"]) & \
          keep_valido(d2["b8a"], d2["b11"], d2["b12"], d2["scl"])
    biom = _bioma_no_grid(GT, H, W)
    dom = val & biom
    R1 = np.stack([refl(d1["red"]), refl(d1["b8a"]), refl(d1["b11"]), refl(d1["b12"])], 0)
    R2 = np.stack([refl(d2["red"]), refl(d2["b8a"]), refl(d2["b11"]), refl(d2["b12"])], 0)
    return R1.astype(np.float32), R2.astype(np.float32), dom, GT, H, W

def _salvar(path, arr, GT, H, W):
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=DST, transform=GT,
                nodata=0, compress="DEFLATE", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    with rasterio.open(path, "w", **prof) as d: d.write(arr.astype(np.uint8), 1)
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", COMPRESS_OVERVIEW="DEFLATE"):
        with rasterio.open(path, "r+") as d: d.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)

def _coords(L):
    c = list(range(0, max(1, L - PATCH + 1), STP))
    if c and c[-1] != L - PATCH and L > PATCH: c.append(L - PATCH)
    return c

def mapa_dl(nome, R1, R2, dom, GT, H, W):
    from redes import construir_dl, MODELOS_DL
    ckp = f"{G.GUATO}/deeplearning/{nome}/{nome}_best.pkl"
    if not os.path.exists(ckp): log(f"[{nome}] sem checkpoint — pulando"); return None
    ck = torch.load(ckp, map_location=DEV, weights_only=False); bandas = ck["bandas"]
    m = construir_dl(nome, p=0.0, pretrained=False).to(DEV); m.load_state_dict(ck["state_dict"]); m.eval()
    mean = ck["mean"][:, None, None]; std = ck["std"][:, None, None]
    X = np.concatenate([R1[bandas], R2[bandas]], 0); X = (X - mean) / std; X *= dom[None]
    prob = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    wins = [(r0, c0) for r0 in _coords(H) for c0 in _coords(W) if dom[r0:r0+PATCH, c0:c0+PATCH].any()]
    with torch.no_grad():
        for i in range(0, len(wins), BATCH):
            lot = wins[i:i+BATCH]
            xb = np.stack([X[:, r0:r0+PATCH, c0:c0+PATCH] for r0, c0 in lot]).astype(np.float32)
            with torch.cuda.amp.autocast(): pb = torch.sigmoid(m(torch.from_numpy(xb).to(DEV))).float().cpu().numpy()
            for (r0, c0), pm in zip(lot, pb): prob[r0:r0+PATCH, c0:c0+PATCH] += pm; cnt[r0:r0+PATCH, c0:c0+PATCH] += 1
    burned = (prob / np.maximum(cnt, 1e-6) >= THR) & dom
    burned = filtrar_min_px(burned, MIN_PX) & dom
    _salvar(f"{OUT}/mascara2025_{nome}.tif", burned, GT, H, W)
    del m; torch.cuda.empty_cache()
    log(f"[{nome}] 2025: {burned.sum()*PX_HA:,.0f} ha -> {OUT}/mascara2025_{nome}.tif")
    return burned.sum() * PX_HA

def mapa_ml(nome, R1, R2, dom, GT, H, W):
    import joblib
    pk = f"{G.GUATO}/machine_learning/{nome}/{nome}.pkl"
    if not os.path.exists(pk): log(f"[{nome}] sem pkl — pulando"); return None
    clf = joblib.load(pk)["modelo"]
    nbr1, nsw1, mir1 = indices(R1[1], R1[2], R1[3]); nbr2, nsw2, mir2 = indices(R2[1], R2[2], R2[3])
    feat = np.concatenate([R1, R2, (nbr1-nbr2)[None], (nsw1-nsw2)[None], (mir1-mir2)[None]], 0)  # (11,H,W)
    out = np.zeros((H, W), bool); idx = np.where(dom.reshape(-1))[0]
    F = feat.reshape(11, -1).T  # (H*W,11)
    pred = np.zeros(H * W, np.uint8)
    for i in range(0, len(idx), 2_000_000):   # predição em chunks (SVM é lento)
        sub = idx[i:i+2_000_000]; pred[sub] = clf.predict(F[sub]).astype(np.uint8)
    burned = (pred.reshape(H, W) > 0) & dom
    burned = filtrar_min_px(burned, MIN_PX) & dom
    _salvar(f"{OUT}/mascara2025_{nome}.tif", burned, GT, H, W)
    log(f"[{nome}] 2025: {burned.sum()*PX_HA:,.0f} ha -> {OUT}/mascara2025_{nome}.tif")
    return burned.sum() * PX_HA

def run(modelos=None):
    from redes import MODELOS_DL
    log("carregando mosaico 2025…")
    R1, R2, dom, GT, H, W = _carrega_2025()
    log(f"grid 2025 {H}x{W} | domínio (válido∩bioma) = {dom.sum()*PX_HA:,.0f} ha")
    dls = list(MODELOS_DL); mls = ["rf", "xgboost", "mlp", "svm"]
    sel = modelos or (dls + mls)
    res = {}
    for nm in sel:
        if nm in MODELOS_DL: r = mapa_dl(nm, R1, R2, dom, GT, H, W)
        else: r = mapa_ml(nm, R1, R2, dom, GT, H, W)
        if r is not None: res[nm] = r
    import json; json.dump({k: float(v) for k, v in res.items()}, open(f"{OUT}/resumo_mascaras2025.json", "w"), indent=2)
    log(f"✅ máscaras 2025 por modelo em {OUT} | referência fixa: {G.REF_2025[0]}")
    return res

if __name__ == "__main__":
    run(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
