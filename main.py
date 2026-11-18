#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guato_ba · main.py — fluxo do estudo (CLI por estágio).

Estágios:
  reference   valida referência fixa ∩ bioma (guardrail)
  data        (re)constrói dataset 2024 (SCL_MIN=0.30) + teste 2025 ∩ bioma
  train-ml    treina os 4 ML com os hiperparâmetros de hpo_best.json → .pkl
  train-dl    treino final 1000ép early-stop por modelo DL → .pkl + curva(loss+IoU)
  accumulate  acumulado 2024 TILE-A-TILE (jan→dez) por modelo → GeoTIFF ∩ bioma + cobertura %
  evaluate    métricas patch (val/test 2025) + acumulado vs referência ∩ bioma
  report      consolida tabelas/curvas

GUARDRAIL #1: tudo dentro do Pantanal. Entradas dos modelos SEM água/nuvem (SCL+WIW; inválidos→0).
Rodar com /opt/conda/envs/geospatial/bin/python.
"""
import os, sys, time, json, argparse, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_data_science as G
from geo_data_science import (log, ESC, IGN, PX_HA, RES, DST, SCL_KEEP, WIW_NIR, WIW_SW1,
                              grid_5880, mascara_bioma, carregar_referencia, carregar_npy,
                              metricas_pixel, metricas_artigo, salvar_raster, filtrar_min_px,
                              tiles_pantanal, geom_por_tile, _slot_do_tile, buscar_itens, compor_data,
                              keep_valido, indices, construir_dataset_2024, construir_teste_2025,
                              extrair_features_pixel, _coords)
import numpy as np
GUATO = G.GUATO
DL_DIR = f"{GUATO}/deeplearning"; ML_DIR = f"{GUATO}/machine_learning"
RES_DIR = f"{GUATO}/results"; HPO_DIR = f"{GUATO}/hpo"
for d in (RES_DIR, HPO_DIR): os.makedirs(d, exist_ok=True)
SEED = 42; PATCH = 128; THR = 0.5; MIN_PX = 12

# ============================================================
#  Dados DL (montagem por modelo + normalização em pixels válidos)
# ============================================================
def montar(T1, T2, bandas):
    return np.concatenate([T1[:, bandas], T2[:, bandas]], axis=1).astype(np.float32) / ESC

def norm_stats(X, Y):
    """mean/std por canal usando só pixels válidos (Y!=IGN)."""
    v = (Y != IGN)[:, None, :, :]
    msk = np.broadcast_to(v, X.shape)
    mean = np.array([X[:, c][v[:, 0]].mean() for c in range(X.shape[1])], np.float32)
    std  = np.array([X[:, c][v[:, 0]].std() + 1e-6 for c in range(X.shape[1])], np.float32)
    return mean, std

# ============================================================
#  Treino DL (reusa infra do estudo de épocas)
# ============================================================
def _torch():
    import torch, torch.nn.functional as F
    return torch, F

def treino_dl(nome, hp, epochs, out_dir, tag="final", min_stop=0, patience=80, log_every=10, live=True, split="train"):
    import torch
    from redes import construir_dl, MODELOS_DL, make_loss
    cfg = MODELOS_DL[nome]; bandas = cfg["bandas"]; MIC = cfg["micro"]; BATCH = 256; ACC = max(1, BATCH // MIC)
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    T1, T2, Y, Gg = carregar_npy(split)
    X = montar(T1, T2, bandas); del T1, T2
    # split 70/30 por GRUPO espacial (Gg) — SEM vazamento: nenhum bloco de ~51km fica em treino E val.
    # (idêntico em espírito ao caminho ML, que usa GroupKFold sobre Gg.) Antes era aleatório por patch,
    # o que vazava: os 3 pares tilam a mesma geografia → a mesma cicatriz caía nos dois lados (val IoU fake).
    hb_all = (Y.reshape(len(Y), -1) == 1).any(1)
    _splmask = os.environ.get("GUATO_SPLITMASK")   # sidecar _SPLIT.npy (1=val): split ESPACIAL p/ holdout honesto
    if _splmask and os.path.exists(_splmask):
        isval = np.load(_splmask).astype(bool); assert len(isval) == len(Y), f"splitmask {len(isval)}!={len(Y)}"
        tr, va = np.where(~isval)[0], np.where(isval)[0]
        log(f"  split ESPACIAL ({os.path.basename(_splmask)}): treino={len(tr)} val={len(va)} | +val={hb_all[va].mean()*100:.1f}% +tr={hb_all[tr].mean()*100:.1f}%"
            f" | grupos tr∩val={len(set(Gg[tr]) & set(Gg[va]))} (deve ser 0)")
    else:
        from sklearn.model_selection import GroupShuffleSplit
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED).split(Y, hb_all, groups=Gg))
        log(f"  split por grupo: treino={len(tr)} val={len(va)} | +val={hb_all[va].mean()*100:.1f}% +tr={hb_all[tr].mean()*100:.1f}%"
            f" | grupos tr∩val={len(set(Gg[tr]) & set(Gg[va]))} (deve ser 0)")
    mean, std = norm_stats(X[tr], Y[tr])                # estatísticas só no TREINO (sem vazamento)
    X = (X - mean[None, :, None, None]) / std[None, :, None, None]
    Xtr, Ytr = X[tr], Y[tr]; Xva, Yva = X[va], Y[va]; del X
    pos = int((Ytr == 1).sum()); neg = int((Ytr == 0).sum())
    pw = torch.tensor(min(neg / max(pos, 1), 10.0), device=DEV)
    hb = (Ytr.reshape(len(Ytr), -1) == 1).any(1); ipos = np.where(hb)[0]; ineg = np.where(~hb)[0]
    psz = int(Xtr.shape[-1]); MIC = max(8, int(round(MIC * 16384.0 / (psz * psz))))   # escala micro-batch p/ caber na GPU (128²->micro cheio; 256²->/4)
    if psz > 128: MIC = max(8, MIC // 2)   # margem extra p/ patch grande (evita OOM: 256²->micro ~32)
    # Amostragem do batch. Em TODOS os modos cada patch POSITIVO é visto 1x/época (exposição de
    # fogo constante); o que muda é quantos NEGATIVOS o acompanham.
    #  - default: batch 256 a 50/50 (128 pos + 128 neg) — histórico.
    #  - GUATO_POS_PER_BATCH / GUATO_NEG_PER_BATCH: contagem ABSOLUTA (ex.: 128 pos + 753 neg).
    #    Mantém o sinal forte de fogo por passo e AUMENTA o não-fogo → reduz comissão, prior mais real.
    #  - GUATO_POSFRAC (fração, ou "nat"): mantém batch 256 e varia a proporção.
    _npb = os.environ.get("GUATO_NEG_PER_BATCH")
    if _npb is not None:
        bpb = int(os.environ.get("GUATO_POS_PER_BATCH", "128")); nbpb = int(_npb)
        BATCH = bpb + nbpb
    else:
        BATCH = 256
        _pf = os.environ.get("GUATO_POSFRAC", "0.5")
        POSFRAC = (len(ipos) / max(1, len(ipos) + len(ineg))) if _pf == "nat" else float(_pf)
        bpb = max(1, int(round(BATCH * POSFRAC))); nbpb = max(0, BATCH - bpb)
    nb_batches = max(1, math.ceil(len(ipos) / bpb))
    log(f"  amostragem: {bpb} pos + {nbpb} neg por batch (BATCH={BATCH}, fração_fogo={bpb/BATCH:.3f}, {nb_batches} batches/época; todos os positivos 1x/época)")
    spe = max(1, nb_batches)   # 1 passo do otimizador por batch
    model = construir_dl(nome, p=hp.get("dropout", 0.2), pretrained=True).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.get("lr", 3e-4), weight_decay=1e-4)
    _SEL = os.environ.get("GUATO_SELECT", "loss")   # "iou" => seleção/early-stop/LR pelo IoU de validação (holdout espacial)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=("max" if _SEL == "iou" else "min"), factor=0.5, patience=10, min_lr=hp.get("lr", 3e-4) * 0.01)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = make_loss(tv_b=hp.get("tv_b", 0.7), w_bce=hp.get("w_bce", 0.5))

    _SPECAUG = os.environ.get("GUATO_SPECAUG", "0") == "1"   # augmentação radiométrica/espectral (default OFF)
    def augment(xb, yb):
        xb = xb.copy(); yb = yb.copy()
        if _SPECAUG:   # drift radiométrica sintética (espaço z-score): ganho/offset por banda + ruído
            C = xb.shape[1]
            gain = rng.normal(1.0, 0.08, size=(len(xb), C, 1, 1)).astype(np.float32)
            off = rng.normal(0.0, 0.05, size=(len(xb), C, 1, 1)).astype(np.float32)
            xb = (xb * gain + off + rng.normal(0.0, 0.03, size=xb.shape).astype(np.float32)).astype(np.float32)
        for k in range(len(xb)):
            if rng.random() < 0.5: xb[k] = xb[k, :, ::-1, :]; yb[k] = yb[k, ::-1, :]
            if rng.random() < 0.5: xb[k] = xb[k, :, :, ::-1]; yb[k] = yb[k, :, ::-1]
            r = int(rng.integers(0, 4))
            if r: xb[k] = np.rot90(xb[k], r, axes=(1, 2)); yb[k] = np.rot90(yb[k], r)
        return np.ascontiguousarray(xb, np.float32), np.ascontiguousarray(yb)

    def mask_input(xb, yb):   # entrada sem água/nuvem: zera inválidos (pós z-score)
        return xb * (yb != IGN).unsqueeze(1).float()

    @torch.no_grad()
    def evaluate():
        model.eval(); TP = FP = FN = TN = 0; tl = 0; nb = 0
        for i in range(0, len(Xva), 64):
            xb = torch.from_numpy(Xva[i:i+64]).to(DEV); yb = torch.from_numpy(Yva[i:i+64].astype(np.int16)).to(DEV)
            xb = mask_input(xb, yb)
            with torch.cuda.amp.autocast():
                lo = model(xb); l = loss_fn(lo, yb, pw)
            tl += float(l); nb += 1
            pr = (torch.sigmoid(lo) >= 0.5) & (yb != IGN); gt = (yb == 1); vd = (yb != IGN)
            TP += int((pr & gt).sum()); FP += int((pr & ~gt & vd).sum())
            FN += int((~pr & gt).sum()); TN += int((~pr & (yb == 0)).sum())
        m = metricas_pixel(TP, FP, FN, TN); m["loss"] = tl / max(nb, 1); return m

    best = (-1.0 if _SEL == "iou" else float("inf")); best_ep = 0; best_iou = 0.0; best_loss = float("inf"); hist = []; t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        # batch estratificado SEM reposição (~12.8% queimada): TODOS os patches de treino 1x/época.
        pi = rng.permutation(ipos); ni = rng.permutation(ineg); ep_sum = 0.0; ep_n = 0
        for b in range(nb_batches):
            # negativos com wrap (módulo) p/ nunca faltar quando nbpb é grande (POSFRAC baixo)
            nsel = ni[(np.arange(b*nbpb, (b+1)*nbpb) % len(ni))] if (nbpb > 0 and len(ni) > 0) else np.array([], int)
            blk = np.concatenate([pi[b*bpb:(b+1)*bpb], nsel]).astype(int)
            if len(blk) == 0: continue
            rng.shuffle(blk)
            opt.zero_grad(set_to_none=True); nmic = max(1, math.ceil(len(blk) / MIC))
            for i in range(0, len(blk), MIC):
                idx = blk[i:i+MIC]; xb, yb = augment(Xtr[idx], Ytr[idx])
                xb = torch.from_numpy(xb).to(DEV); yb = torch.from_numpy(yb.astype(np.int16)).to(DEV)
                xb = mask_input(xb, yb)
                with torch.cuda.amp.autocast():
                    lo = model(xb); l = loss_fn(lo, yb, pw) / nmic
                scaler.scale(l).backward(); ep_sum += float(l) * nmic; ep_n += 1
            scaler.step(opt); scaler.update()
        trl = ep_sum / max(ep_n, 1)   # loss MÉDIO de treino na época (curva train vs val -> checar overfitting)
        m = evaluate(); sch.step(m["IoU"] if _SEL == "iou" else m["loss"])   # LR no platô do IoU (iou) ou do val_loss
        hist.append((ep, trl, m["loss"], m["F1"], m["IoU"], m["recall"], opt.param_groups[0]["lr"]))
        import pandas as pd
        pd.DataFrame(hist, columns=["epoca", "treino_loss", "val_loss", "val_f1", "val_iou", "val_recall", "lr"]).to_csv(f"{out_dir}/historico.csv", index=False)
        cur = m["IoU"] if _SEL == "iou" else m["loss"]
        improved = (cur > best) if _SEL == "iou" else (cur < best)
        if improved:   # seleciona pelo MAIOR IoU de val (iou) ou MENOR val_loss (loss)
            best = cur; best_ep = ep; best_iou = m["IoU"]; best_loss = m["loss"]
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "in_ch": cfg["in_ch"],
                        "bandas": bandas, "epoca": ep, "dropout": hp.get("dropout", 0.2), "modelo": nome, "hp": hp},
                       f"{out_dir}/{nome}_best.pkl")
        if live and (ep % 5 == 0 or ep <= 3): _curva(hist, nome, out_dir, epochs)
        if ep % log_every == 0:
            log(f"  [{nome}] ep{ep}/{epochs} trLoss={trl:.4f} vLoss={m['loss']:.4f} gap={m['loss']-trl:+.4f} vIoU={m['IoU']:.4f} best@{best_ep}(IoU={best_iou:.3f},loss={best_loss:.4f}) lr={opt.param_groups[0]['lr']:.1e} ({(time.time()-t0)/ep:.1f}s/ep)")
        if min_stop and ep >= min_stop and (ep - best_ep) >= patience:
            log(f"  [{nome}] early-stop ep{ep} (best@{best_ep}: IoU={best_iou:.3f} loss={best_loss:.4f})"); break
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "in_ch": cfg["in_ch"],
                "bandas": bandas, "epoca": ep, "dropout": hp.get("dropout", 0.2), "modelo": nome, "hp": hp},
               f"{out_dir}/{nome}_last.pkl")
    if live: _curva(hist, nome, out_dir, epochs)
    return best, best_ep, hist

def _curva(hist, nome, out_dir, epmax):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = np.array(hist)
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(h[:, 0], h[:, 1], color="#2ca02c", lw=1.2, label="treino (loss)")
    ax1.plot(h[:, 0], h[:, 2], color="#d62728", lw=1.2, label="val (loss)")
    ax1.set_xlabel("época"); ax1.set_ylabel("loss"); ax1.grid(ls=":", alpha=.5)
    ax2 = ax1.twinx()
    ax2.plot(h[:, 0], h[:, 4], color="#1f77b4", ls="--", lw=1.6, label="val IoU")
    ax2.plot(h[:, 0], h[:, 3], color="#9467bd", ls=":", lw=1.4, label="val F1")
    ax2.set_ylabel("val IoU / F1"); ax2.set_ylim(0, 1)
    ax1.set_title(f"{nome} — ép {int(h[-1,0])}/{epmax} | best IoU={float(h[:,4].max()):.4f} | val F1={float(h[-1,3]):.4f}")
    l1, b1 = ax1.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out_dir}/curva.png", dpi=110); plt.close(fig)

# ============================================================
#  Treino ML (hiperparâmetros fixos de hpo_best.json — sem Optuna)
# ============================================================
def _feats_split():
    cache = {k: f"{G.DATA_DIR}/feat_{k}.npy" for k in ("Xtr", "ytr", "Xva", "yva")}
    if all(os.path.exists(v) for v in cache.values()):
        return (np.load(cache["Xtr"]), np.load(cache["ytr"]), np.load(cache["Xva"]), np.load(cache["yva"]))
    T1, T2, Y, Gg = carregar_npy("train")
    from sklearn.model_selection import GroupKFold
    tr, va = list(GroupKFold(5).split(T1, Y, Gg))[0]
    Xtr, ytr = extrair_features_pixel(T1[tr], T2[tr], Y[tr])
    Xva, yva = extrair_features_pixel(T1[va], T2[va], Y[va], n_por_classe=120000)
    np.save(cache["Xtr"], Xtr); np.save(cache["ytr"], ytr); np.save(cache["Xva"], Xva); np.save(cache["yva"], yva)
    log(f"features: treino {Xtr.shape} (pos {int(ytr.sum())}) | val {Xva.shape}")
    return Xtr, ytr, Xva, yva

def _ml_kwargs(nome, params):
    """Converte os best_params salvos (hpo_best.json) nos kwargs reais do modelo (trata mlp)."""
    bp = dict(params)
    if nome == "mlp":
        if "hidden" in bp: bp["hidden_layer_sizes"] = tuple(int(z) for z in bp.pop("hidden").split(","))
        if "lr" in bp: bp["learning_rate_init"] = bp.pop("lr")
    return bp

def train_ml(nome):
    """Treina um modelo ML com os hiperparâmetros fixos de machine_learning/<nome>/hpo_best.json
    (sem busca Optuna). Se o arquivo não existir, usa a configuração padrão da fábrica."""
    import joblib
    from redes import MODELOS_ML
    od = f"{ML_DIR}/{nome}"
    if os.path.exists(f"{od}/{nome}.pkl"):
        log(f"[ML {nome}] já treinado — pulando"); return
    hb = f"{od}/hpo_best.json"
    params = json.load(open(hb)).get("best_params", {}) if os.path.exists(hb) else {}
    Xtr, ytr, Xva, yva = _feats_split()
    if nome == "svm":   # SVM é O(n^2): limita amostras
        rng = np.random.default_rng(SEED)
        st_ = rng.choice(len(Xtr), min(40000, len(Xtr)), replace=False); Xtr, ytr = Xtr[st_], ytr[st_]
        sv_ = rng.choice(len(Xva), min(20000, len(Xva)), replace=False); Xva, yva = Xva[sv_], yva[sv_]
    clf = MODELOS_ML[nome](**_ml_kwargs(nome, params))
    clf.fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
    os.makedirs(od, exist_ok=True)
    joblib.dump({"modelo": clf, "nome": nome, "features": 11, "hp": params}, f"{od}/{nome}.pkl")
    log(f"[ML {nome}] treinado ({'hpo_best.json' if params else 'config padrão'}) → {od}/{nome}.pkl")

# ============================================================
#  Avaliação patch (val 2024 + teste 2025)
# ============================================================
def avaliar_patch():
    import torch, joblib, pandas as pd
    from redes import construir_dl, MODELOS_DL
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    # DL
    for nome in MODELOS_DL:
        ckp = f"{DL_DIR}/{nome}/{nome}_best.pkl"
        if not os.path.exists(ckp): continue
        ck = torch.load(ckp, map_location=DEV, weights_only=False); bandas = ck["bandas"]
        m = construir_dl(nome, p=ck.get("dropout", 0.0), pretrained=False).to(DEV); m.load_state_dict(ck["state_dict"]); m.eval()
        mean = ck["mean"][None, :, None, None]; std = ck["std"][None, :, None, None]
        for split in ("val", "test2025"):
            T1, T2, Y, Gg = carregar_npy("train" if split == "val" else "test2025")
            if split == "val":
                from sklearn.model_selection import GroupKFold
                _, va = list(GroupKFold(5).split(T1, Y, Gg))[0]; T1, T2, Y = T1[va], T2[va], Y[va]
            X = (montar(T1, T2, bandas) - mean) / std
            TP = FP = FN = TN = 0
            with torch.no_grad():
                for i in range(0, len(X), 64):
                    xb = torch.from_numpy(X[i:i+64]).to(DEV); yb = torch.from_numpy(Y[i:i+64].astype(np.int16)).to(DEV)
                    xb = xb * (yb != IGN).unsqueeze(1).float()
                    with torch.cuda.amp.autocast(): pr = (torch.sigmoid(m(xb)) >= THR)
                    vd = (yb != IGN); gt = (yb == 1)
                    TP += int((pr & gt).sum()); FP += int((pr & ~gt & vd).sum()); FN += int((~pr & gt).sum()); TN += int((~pr & (yb == 0)).sum())
            mm = metricas_pixel(TP, FP, FN, TN); mm.update(modelo=nome, tipo="DL", split=split); rows.append(mm)
            log(f"  [{nome}/{split}] IoU={mm['IoU']:.4f} F1={mm['F1']:.4f} OE={mm['OE']:.3f} CE={mm['CE']:.3f}")
        del m; torch.cuda.empty_cache()
    # ML
    for nome in ("rf", "xgboost", "mlp", "svm"):
        pk = f"{ML_DIR}/{nome}/{nome}.pkl"
        if not os.path.exists(pk): continue
        clf = joblib.load(pk)["modelo"]
        for split in ("val", "test2025"):
            T1, T2, Y, Gg = carregar_npy("train" if split == "val" else "test2025")
            if split == "val":
                from sklearn.model_selection import GroupKFold
                _, va = list(GroupKFold(5).split(T1, Y, Gg))[0]; T1, T2, Y = T1[va], T2[va], Y[va]
            Xf, yf = extrair_features_pixel(T1, T2, Y, n_por_classe=200000, balancear=False)  # proporção natural (realista)
            p = clf.predict(Xf)
            TP = int(((p == 1) & (yf == 1)).sum()); FP = int(((p == 1) & (yf == 0)).sum())
            FN = int(((p == 0) & (yf == 1)).sum()); TN = int(((p == 0) & (yf == 0)).sum())
            mm = metricas_pixel(TP, FP, FN, TN); mm.update(modelo=nome, tipo="ML", split=split); rows.append(mm)
            log(f"  [{nome}/{split}] IoU={mm['IoU']:.4f} F1={mm['F1']:.4f} OE={mm['OE']:.3f} CE={mm['CE']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES_DIR}/tabela_patch.csv", index=False)
    log(f"✅ tabela_patch → {RES_DIR}/tabela_patch.csv"); print(df.to_string(index=False))

# ============================================================
#  CLI
# ============================================================
def comparar_acumulado(modelo):
    """Compara o acumulado 2024 de um modelo vs referência fixa ∩ bioma (+ DETER/MapBiomas),
    nos domínios pantanal_inteiro e analisado_SCL; reporta a % do Pantanal analisada."""
    import rasterio, pandas as pd
    from geo_data_science import carregar_deter, carregar_mapbiomas
    ACU = f"{G.GUATO}/results/acumulado"
    with rasterio.open(f"{ACU}/acum2024_{modelo}.tif") as s: MOD = s.read(1) > 0
    with rasterio.open(f"{ACU}/analisado_{modelo}.tif") as s: ANAL = s.read(1) > 0
    B = mascara_bioma(); REF = carregar_referencia(2024)
    cob = ANAL.sum() / max(B.sum(), 1) * 100
    log(f"[{modelo}] cobertura do Pantanal analisada = {cob:.1f}% | modelo={MOD.sum()*PX_HA:,.0f}ha | ref={REF.sum()*PX_HA:,.0f}ha")
    refs = {"referencia": REF}
    try: refs["DETER"] = carregar_deter(2024)
    except Exception: pass
    try: refs["MapBiomas"] = carregar_mapbiomas()
    except Exception: pass
    doms = {"pantanal_inteiro": B, "analisado_SCL": ANAL & B}
    rows = []
    for dn, dom in doms.items():
        for rn, A in refs.items():
            m = metricas_artigo(MOD, A, dom); m.update(modelo=modelo, dominio=dn, referencia=rn, cobertura_pct=cob); rows.append(m)
    df = pd.DataFrame(rows)[["modelo", "dominio", "referencia", "cobertura_pct", "F1", "IoU", "OE", "CE", "BIAS",
                             "area_modelo_ha", "area_ref_ha", "FP_ha", "FN_ha"]]
    out = f"{RES_DIR}/tabela_acumulado_{modelo}.csv"; df.to_csv(out, index=False)
    import pandas as pd2; pd2.set_option("display.width", 220); pd2.set_option("display.float_format", lambda x: f"{x:.3f}")
    log(f"✅ {out}"); print(df.to_string(index=False))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["reference", "data", "train-ml", "train-dl",
                                       "accumulate", "accumulate-full", "eval-accum", "evaluate", "mapa-2025", "report"])
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--min_stop", type=int, default=200)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--tiles", default=None, help="lista de tiles, ex: 21KWB,21KVB")
    ap.add_argument("--cloud", type=float, default=10.0)      # config ADOTADA: nuvem<=10% na cena
    ap.add_argument("--in_bioma", type=float, default=0.95)   # <=5% de nuvem dentro do Pantanal (area valida>=95%)
    ap.add_argument("--thr", type=float, default=0.90)
    ap.add_argument("--mmu", type=int, default=24)            # ~1 ha
    ap.add_argument("--scl", default="4,5,7", help="classes SCL válidas, ex: 4,5 ou 4,5,7")
    ap.add_argument("--tag", default="")
    ap.add_argument("--forcar", action="store_true")
    a = ap.parse_args()

    if a.stage == "reference":
        B = mascara_bioma(); ref = carregar_referencia(2024)
        log(f"referência 2024 ∩ bioma = {ref.sum()*PX_HA:,.0f} ha | fora bioma = {int((ref & ~B).sum())} (==0 ok)")

    elif a.stage == "data":
        construir_dataset_2024(forcar=a.forcar); construir_teste_2025(forcar=a.forcar)

    elif a.stage == "train-ml":
        mods = [a.modelo] if a.modelo else ["rf", "xgboost", "mlp", "svm"]
        for nm in mods: train_ml(nm)

    elif a.stage == "train-dl":
        from redes import MODELOS_DL
        mods = [a.modelo] if a.modelo else list(MODELOS_DL)
        for nm in mods:
            od = f"{DL_DIR}/{nm}"; os.makedirs(od, exist_ok=True)
            hp = {}
            hb = f"{od}/hpo_best.json"
            if os.path.exists(hb): hp = json.load(open(hb)).get("best_params", {})
            log(f"=== train-dl {nm} (hp={hp or 'default'}) ===")
            best, bep, _ = treino_dl(nm, hp, a.epochs, od, tag="final",
                                     min_stop=min(a.min_stop, a.epochs), patience=a.patience)
            log(f"=== {nm}: best val IoU={best:.4f}@{bep} → {od} ===")

    elif a.stage == "accumulate":
        import accumulate_tile as AT
        tiles = a.tiles.split(",") if a.tiles else None
        scl = tuple(int(x) for x in a.scl.split(","))
        AT.run(a.modelo or "biatt_unet", tiles=tiles, cloud=a.cloud, in_bioma_min=a.in_bioma,
               thr=a.thr, mmu_px=a.mmu, tag=a.tag, scl_keep=scl)

    elif a.stage == "accumulate-full":
        import accumulate_tile as AT
        scl = tuple(int(x) for x in a.scl.split(","))
        AT.run_pantanal(a.modelo or "biatt_unet", cloud=a.cloud, in_bioma_min=a.in_bioma,
                        thr=a.thr, mmu_px=a.mmu, tag=a.tag or "_pantanal", scl_keep=scl)

    elif a.stage == "eval-accum":
        from redes import MODELOS_DL
        mods = [a.modelo] if a.modelo else [m for m in MODELOS_DL if os.path.exists(f"{G.GUATO}/results/acumulado/acum2024_{m}.tif")]
        for nm in mods: comparar_acumulado(nm)

    elif a.stage == "mapa-2025":
        sys.path.insert(0, f"{G.GUATO}/scripts/figuras")   # script realocado em scripts/figuras/
        import mapa_2025
        mapa_2025.run(a.modelo.split(",") if a.modelo else None)

    elif a.stage == "evaluate":
        avaliar_patch()

    elif a.stage == "report":
        log("report: ver results/ (tabela_patch.csv, tabela_acumulado.csv, curvas por modelo)")

if __name__ == "__main__":
    main()
