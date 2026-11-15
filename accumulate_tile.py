#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guato_ba · accumulate_tile.py — ACUMULADO 2024 TILE-A-TILE (um tile por vez, sem mosaico global).

Objetivo: recobrir o Pantanal inteiro. Fluxo por tile (ano inteiro jan→dez):
  1) todas as cenas do tile com <30% de nuvem;
  2) recorta SCL ao Pantanal; une revisitas do mesmo dia (refl=média, validade=união);
  3) filtra datas cujo SCL válido cobre >50% do Pantanal no tile;
  4) pares consecutivos; modelo só em pixels válidos (SCL+WIW; entrada zerada nos inválidos).
Acumula no grid global 5880; ao final clip ao bioma + sieve ≥12px. Reporta a % do Pantanal analisada.
GUARDRAIL: tudo dentro do Pantanal.
"""
import os, time, math, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from rasterio.features import rasterize, shapes as rio_shapes
import geo_data_science as G
from geo_data_science import (log, grid_5880, mascara_bioma, geom_por_tile, _slot_do_tile,
                              buscar_itens, DST, RES, PX_HA, salvar_raster, filtrar_min_px)
from redes import construir_dl, MODELOS_DL

# Config ADOTADA (campeã): cloud<=10% na cena, <=5% de nuvem dentro do Pantanal (area valida>=95%),
# THR=0.90, MMU=24px (~1ha). Ver estudo 21KWB: joelho em area>=95% (IoU 0.740); >=97% colapsa a cobertura temporal.
INI = "2024-01-01"; FIM = "2024-12-31"; CLOUD = 10.0; IN_BIOMA_MIN = 0.95
PATCH = 128; OV = 16; STP = PATCH - OV; BATCH = 64; THR = 0.90; MIN_PX = 24
NPROC = 5   # processos p/ compor datas em paralelo (I/O de COG; seguro, sem segfault de thread do GDAL)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACU_DIR = f"{G.GUATO}/results/acumulado"; os.makedirs(ACU_DIR, exist_ok=True)

# ---------- worker (processo) que compõe UMA data: lê+warpa as cenas, une revisitas (média) ----------
def _compose_worker(payload):
    import os as _os
    _os.environ["PROJ_DATA"] = "/opt/conda/envs/geospatial/share/proj"
    _os.environ["PROJ_LIB"] = "/opt/conda/envs/geospatial/share/proj"
    _os.environ["GDAL_DATA"] = "/opt/conda/envs/geospatial/share/gdal"
    import numpy as _np, rasterio
    from rasterio import Affine as _Aff
    from rasterio.crs import CRS as _CRS
    from rasterio.warp import reproject as _rp, Resampling as _Rs
    scenes, st_t, sh, sw, scl_keep = payload
    st = _Aff(*st_t); dst = _CRS.from_epsg(5880)
    SCL_KEEP = tuple(scl_keep); WIW_NIR = 0.1804; WIW_SW1 = 0.1131; ESC = 10000.0
    def rb(href):
        o = _np.zeros((sh, sw), _np.uint16)
        with rasterio.open(href) as s:
            _rp(rasterio.band(s, 1), o, src_transform=s.transform, src_crs=s.crs,
                dst_transform=st, dst_crs=dst, resampling=_Rs.nearest)
        return o
    sumR = _np.zeros((4, sh, sw), _np.float32); cnt = _np.zeros((sh, sw), _np.float32); cval = _np.zeros((sh, sw), bool)
    for sc in scenes:
        try:
            red = rb(sc["B04"]); n = rb(sc["B8A"]); s1 = rb(sc["B11"]); s2 = rb(sc["B12"]); scl = rb(sc["SCL"])
        except Exception:
            continue
        val = (n > 0) & (s1 > 0) & (s2 > 0) & _np.isin(scl, SCL_KEEP) & ~(((n / ESC) < WIW_NIR) & ((s1 / ESC) < WIW_SW1))
        if not val.any():
            continue
        R = _np.stack([red / ESC, n / ESC, s1 / ESC, s2 / ESC], 0).astype(_np.float32)
        sumR[:, val] += R[:, val]; cnt[val] += 1; cval |= val
    return (sumR / _np.maximum(cnt, 1)).astype(_np.float32), cval

def _coords(L):
    c = list(range(0, max(1, L - PATCH + 1), STP))
    if c and c[-1] != L - PATCH and L > PATCH: c.append(L - PATCH)
    return c

def _infere_par(model, bandas, mean, std, Ra, Rb, dom, sh, sw):
    """Inferência janela deslizante de um par. Entrada mascarada (inválidos→0). Retorna prob média."""
    X = np.concatenate([Ra[bandas], Rb[bandas]], 0).astype(np.float32)   # (C,sh,sw) já reflectância
    X = (X - mean[:, None, None]) / std[:, None, None]
    X *= dom[None]                                                       # zera fora do domínio (sem água/nuvem)
    prob = np.zeros((sh, sw), np.float32); cnt = np.zeros((sh, sw), np.float32)
    wins = [(r0, c0) for r0 in _coords(sh) for c0 in _coords(sw) if dom[r0:r0+PATCH, c0:c0+PATCH].any()]
    with torch.no_grad():
        for i in range(0, len(wins), BATCH):
            lot = wins[i:i+BATCH]
            xb = np.stack([X[:, r0:r0+PATCH, c0:c0+PATCH] for r0, c0 in lot]).astype(np.float32)
            with torch.cuda.amp.autocast():
                pb = torch.sigmoid(model(torch.from_numpy(xb).to(DEV))).float().cpu().numpy()
            for (r0, c0), pm in zip(lot, pb):
                prob[r0:r0+PATCH, c0:c0+PATCH] += pm; cnt[r0:r0+PATCH, c0:c0+PATCH] += 1
    return prob / np.maximum(cnt, 1e-6)

def _doy(dstr):
    return datetime.strptime(dstr, "%Y-%m-%d").timetuple().tm_yday

def _processar_tile(model, bandas, mean, std, tile, geom, cloud, in_bioma_min, thr, scl_keep, pool):
    """Processa UM tile (jan->dez): baixa cenas, compõe datas, filtra por cobertura limpa,
    roda pares consecutivos. Retorna dict (slot + arrays locais) ou None se vazio/sem dados."""
    r0, c0, sh, sw, st = _slot_do_tile(geom)
    if sh <= 0 or sw <= 0: return None
    tile_mask = rasterize([(geom, 1)], out_shape=(sh, sw), transform=st, fill=0, dtype="uint8").astype(bool)
    area_tile = int(tile_mask.sum())
    if area_tile == 0: return None
    try:
        itens = buscar_itens(tile, INI, FIM, cloud)
    except Exception as e:
        log(f"[{tile}] STAC erro: {e}"); return None
    pordia = {}
    for it in itens:
        dia = it.properties["datetime"][:10]; pordia.setdefault(dia, []).append(it)
    st_t = (st.a, st.b, st.c, st.d, st.e, st.f)
    dias = sorted(pordia)
    payloads = [([{b: it.assets[b].href for b in ("B04", "B8A", "B11", "B12", "SCL") if b in it.assets}
                  for it in pordia[dia]], st_t, sh, sw, scl_keep) for dia in dias]
    comp = {}; ncenas = 0
    for dia, (Rr, val) in zip(dias, pool.map(_compose_worker, payloads)):
        ncenas += len(pordia[dia])
        vt = val & tile_mask
        if area_tile and vt.sum() / area_tile >= in_bioma_min:
            comp[dia] = (Rr, vt)
    ds = sorted(comp)
    burned_tile = np.zeros((sh, sw), bool); anal_tile = np.zeros((sh, sw), bool)
    prob_tile = np.zeros((sh, sw), np.float32); det_tile = np.zeros((sh, sw), np.uint16)
    for k in range(len(ds) - 1):
        Ra, va = comp[ds[k]]; Rb, vb = comp[ds[k+1]]
        dom = va & vb & tile_mask
        if not dom.any(): continue
        prob = _infere_par(model, bandas, mean, std, Ra, Rb, dom, sh, sw)
        det = (prob >= thr) & dom
        novo = det & (det_tile == 0)                 # 1ª vez que esse pixel aparece queimado
        det_tile[novo] = _doy(ds[k+1])               # DOY da data t2 do par (confirmação)
        burned_tile |= det
        prob_tile = np.maximum(prob_tile, np.where(dom, prob, 0.0))
        anal_tile |= dom
    return dict(r0=r0, c0=c0, sh=sh, sw=sw, tile_mask=tile_mask, area_tile=area_tile,
                burned=(burned_tile & tile_mask), anal=(anal_tile & tile_mask),
                prob=((prob_tile * 255).astype(np.uint8) * tile_mask),
                det=np.where(tile_mask, det_tile, 0).astype(np.uint16),
                ncenas=ncenas, ndatas=len(ds))

def _merge_tile(res, ACUM, ANALISADO, PROB, DET):
    """Insere o resultado local de um tile nas matrizes globais (união; DOY mais antigo por 1º a chegar)."""
    r0, c0, sh, sw = res["r0"], res["c0"], res["sh"], res["sw"]; tm = res["tile_mask"]
    ACUM[r0:r0+sh, c0:c0+sw] |= res["burned"]
    ANALISADO[r0:r0+sh, c0:c0+sw] |= res["anal"]
    psub = PROB[r0:r0+sh, c0:c0+sw]; PROB[r0:r0+sh, c0:c0+sw] = np.maximum(psub, res["prob"])
    dsub = DET[r0:r0+sh, c0:c0+sw]; m0 = (dsub == 0) & tm; dsub[m0] = res["det"][m0]

def run(modelo, tiles=None, cloud=CLOUD, in_bioma_min=IN_BIOMA_MIN, thr=THR, mmu_px=MIN_PX, tag="", scl_keep=(4, 5, 7), ckpt=None):
    """Acumulado tile-a-tile. tiles=lista p/ subconjunto; cloud/in_bioma_min ajustáveis; tag = sufixo nos arquivos.
    ckpt=caminho .pkl específico (ablação); default=deeplearning/<modelo>/<modelo>_best.pkl.
    Rastreia a DOY da 1ª detecção por pixel -> atributo data_deteccao no GPKG."""
    assert modelo in MODELOS_DL, f"modelo DL inválido: {modelo}"
    ckp = ckpt or f"{G.GUATO}/deeplearning/{modelo}/{modelo}_best.pkl"
    if not os.path.exists(ckp):
        raise FileNotFoundError(f"checkpoint ausente: {ckp}")
    ck = torch.load(ckp, map_location=DEV, weights_only=False); bandas = ck["bandas"]
    mean = ck["mean"]; std = ck["std"]
    model = construir_dl(modelo, p=0.0, pretrained=False).to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    GH, GW, GGT = grid_5880(); BIOMA = mascara_bioma()
    ACUM = np.zeros((GH, GW), bool); ANALISADO = np.zeros((GH, GW), bool)
    PROB = np.zeros((GH, GW), np.uint8); DET = np.zeros((GH, GW), np.uint16)   # prob×255 e DOY da 1ª detecção
    geoms = geom_por_tile(); alltiles = sorted(geoms)
    sel = [t for t in alltiles if (tiles is None or t in tiles)]
    log(f"acumulado {modelo} | tiles={sel if tiles else 'todos'} | cloud<{cloud}% | area_limpa>={in_bioma_min:.0%} | THR={thr} | MMU={mmu_px}px | SCL={scl_keep}")
    pool = ProcessPoolExecutor(max_workers=NPROC)
    for ti, tile in enumerate(sel):
        t0 = time.time()
        res = _processar_tile(model, bandas, mean, std, tile, geoms[tile], cloud, in_bioma_min, thr, scl_keep, pool)
        if res is None:
            log(f"[{tile}] vazio/sem dados ({ti+1}/{len(sel)})"); continue
        _merge_tile(res, ACUM, ANALISADO, PROB, DET)
        log(f"[{tile}] {res['ncenas']} cenas<{cloud}% | {res['ndatas']} datas limpas(>= {in_bioma_min:.0%}) | queimado {res['burned'].sum()*PX_HA:,.0f}ha | analisado {res['anal'].sum()/max(res['area_tile'],1)*100:.0f}% ({ti+1}/{len(sel)}, {time.time()-t0:.0f}s)")
    pool.shutdown(wait=True)
    ACUM &= BIOMA; ANALISADO &= BIOMA
    ACUM = filtrar_min_px(ACUM, mmu_px) & BIOMA
    DET[~ACUM] = 0
    salvar_raster(f"{ACU_DIR}/acum2024_{modelo}{tag}.tif", ACUM.astype(np.uint8))
    salvar_raster(f"{ACU_DIR}/analisado_{modelo}{tag}.tif", ANALISADO.astype(np.uint8))
    PROB[~BIOMA] = 0; salvar_raster(f"{ACU_DIR}/prob2024_{modelo}{tag}.tif", PROB, nodata=0, dtype="uint8")
    salvar_raster(f"{ACU_DIR}/detdoy_{modelo}{tag}.tif", DET, nodata=0, dtype="uint16")
    _gpkg_com_data(ACUM, DET, GGT, f"{ACU_DIR}/acum2024_{modelo}{tag}.gpkg", mmu_px)
    cob = ANALISADO.sum() / max(BIOMA.sum(), 1) * 100
    resumo = dict(modelo=modelo, tag=tag, cloud=cloud, area_limpa_min=in_bioma_min, thr=thr, mmu_px=mmu_px,
                  area_queimada_ha=float(ACUM.sum()*PX_HA), cobertura_pantanal_pct=float(cob))
    json.dump(resumo, open(f"{ACU_DIR}/resumo_{modelo}{tag}.json", "w"), indent=2)
    log(f"✅ ACUMULADO {modelo}{tag}: {ACUM.sum()*PX_HA:,.0f} ha | cobertura {cob:.1f}% → {ACU_DIR}")
    return resumo

def run_pantanal(modelo="biatt_unet", cloud=CLOUD, in_bioma_min=IN_BIOMA_MIN, thr=THR, mmu_px=MIN_PX,
                 scl_keep=(4, 5, 7), tag="_pantanal", ckpt=None):
    """Acumulado do PANTANAL INTEIRO, TILE A TILE com checkpoint por tile (retomável),
    e no fim DISSOLVE de todos os tiles num único GPKG. Processa um tile por vez; salva um
    .npz por tile em results/acumulado/_ckpt_<modelo><tag>/ e, ao reiniciar, reaproveita os prontos.
    ckpt=caminho .pkl específico (ex. variante de ablação); default = deeplearning/<modelo>/<modelo>_best.pkl."""
    assert modelo in MODELOS_DL, f"modelo DL inválido: {modelo}"
    ckp = ckpt or f"{G.GUATO}/deeplearning/{modelo}/{modelo}_best.pkl"
    if not os.path.exists(ckp):
        raise FileNotFoundError(f"checkpoint ausente: {ckp}")
    ck = torch.load(ckp, map_location=DEV, weights_only=False); bandas = ck["bandas"]
    mean = ck["mean"]; std = ck["std"]
    model = construir_dl(modelo, p=0.0, pretrained=False).to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    GH, GW, GGT = grid_5880(); BIOMA = mascara_bioma()
    ACUM = np.zeros((GH, GW), bool); ANALISADO = np.zeros((GH, GW), bool)
    PROB = np.zeros((GH, GW), np.uint8); DET = np.zeros((GH, GW), np.uint16)
    geoms = geom_por_tile(); alltiles = sorted(geoms)
    CKDIR = f"{ACU_DIR}/_ckpt_{modelo}{tag}"; os.makedirs(CKDIR, exist_ok=True)
    log(f"PANTANAL {modelo}{tag} | {len(alltiles)} tiles | cloud<{cloud}% | area_limpa>={in_bioma_min:.0%} | THR={thr} | MMU={mmu_px}px | SCL={scl_keep} | ckpt={CKDIR}")
    pool = ProcessPoolExecutor(max_workers=NPROC)
    t_all = time.time()
    for ti, tile in enumerate(alltiles):
        ckf = f"{CKDIR}/{tile}.npz"; t0 = time.time()
        if os.path.exists(ckf):
            z = np.load(ckf)
            if int(z["sh"]) == 0:
                log(f"[{tile}] ckpt vazio reaproveitado ({ti+1}/{len(alltiles)})"); continue
            res = dict(r0=int(z["r0"]), c0=int(z["c0"]), sh=int(z["sh"]), sw=int(z["sw"]),
                       tile_mask=z["tile_mask"], burned=z["burned"], anal=z["anal"], prob=z["prob"], det=z["det"])
            _merge_tile(res, ACUM, ANALISADO, PROB, DET)
            log(f"[{tile}] ckpt reaproveitado | queimado {res['burned'].sum()*PX_HA:,.0f}ha ({ti+1}/{len(alltiles)})"); continue
        res = _processar_tile(model, bandas, mean, std, tile, geoms[tile], cloud, in_bioma_min, thr, scl_keep, pool)
        if res is None:
            np.savez_compressed(ckf, r0=0, c0=0, sh=0, sw=0)
            log(f"[{tile}] vazio/sem dados ({ti+1}/{len(alltiles)})"); continue
        np.savez_compressed(ckf, r0=res["r0"], c0=res["c0"], sh=res["sh"], sw=res["sw"],
                            tile_mask=res["tile_mask"], burned=res["burned"], anal=res["anal"],
                            prob=res["prob"], det=res["det"])
        _merge_tile(res, ACUM, ANALISADO, PROB, DET)
        log(f"[{tile}] {res['ncenas']} cenas<{cloud}% | {res['ndatas']} datas limpas(>= {in_bioma_min:.0%}) | queimado {res['burned'].sum()*PX_HA:,.0f}ha | analisado {res['anal'].sum()/max(res['area_tile'],1)*100:.0f}% ({ti+1}/{len(alltiles)}, {time.time()-t0:.0f}s)")
    pool.shutdown(wait=True)
    # ---- DISSOLVE de todos os tiles (grid global) ----
    ACUM &= BIOMA; ANALISADO &= BIOMA
    ACUM = filtrar_min_px(ACUM, mmu_px) & BIOMA
    DET[~ACUM] = 0
    salvar_raster(f"{ACU_DIR}/acum2024_{modelo}{tag}.tif", ACUM.astype(np.uint8))
    salvar_raster(f"{ACU_DIR}/analisado_{modelo}{tag}.tif", ANALISADO.astype(np.uint8))
    PROB[~BIOMA] = 0; salvar_raster(f"{ACU_DIR}/prob2024_{modelo}{tag}.tif", PROB, nodata=0, dtype="uint8")
    salvar_raster(f"{ACU_DIR}/detdoy_{modelo}{tag}.tif", DET, nodata=0, dtype="uint16")
    _gpkg_com_data(ACUM, DET, GGT, f"{ACU_DIR}/acum2024_{modelo}{tag}.gpkg", mmu_px)
    cob = ANALISADO.sum() / max(BIOMA.sum(), 1) * 100
    resumo = dict(modelo=modelo, tag=tag, cloud=cloud, area_limpa_min=in_bioma_min, thr=thr, mmu_px=mmu_px,
                  area_queimada_ha=float(ACUM.sum()*PX_HA), cobertura_pantanal_pct=float(cob),
                  n_tiles=len(alltiles), tempo_min=round((time.time()-t_all)/60, 1))
    json.dump(resumo, open(f"{ACU_DIR}/resumo_{modelo}{tag}.json", "w"), indent=2)
    log(f"✅ PANTANAL {modelo}{tag}: {ACUM.sum()*PX_HA:,.0f} ha | cobertura {cob:.1f}% | {len(alltiles)} tiles em {(time.time()-t_all)/60:.0f}min → {ACU_DIR}")
    return resumo

def _gpkg_com_data(ACUM, DET, GGT, path, mmu_px):
    """GPKG dos polígonos queimados. Por POLÍGONO COMPLETO (após touch/fusão): area_ha,
    data_inicial (DOY mais antigo), data_final (DOY mais recente), duracao_dias, mes_inicial."""
    import geopandas as gpd
    from datetime import timedelta
    from scipy import ndimage
    from shapely.geometry import shape
    S8 = np.ones((3, 3), int)
    lab, n = ndimage.label(ACUM, structure=S8)
    if n == 0:
        log("sem polígonos p/ gpkg"); return
    idx = np.arange(1, n + 1)
    doy_min = ndimage.minimum(np.where(DET == 0, 9999, DET), lab, index=idx)   # data mais antiga (ignora 0)
    doy_max = ndimage.maximum(DET, lab, index=idx)                             # data mais recente
    geoms, labs = [], []
    for geom, v in rio_shapes(lab, mask=ACUM, transform=GGT, connectivity=8):
        geoms.append(shape(geom)); labs.append(int(v))
    gdf = gpd.GeoDataFrame({"lab": labs}, geometry=geoms, crs="EPSG:5880")
    gdf = gdf.dissolve(by="lab").reset_index()      # garante 1 feição por polígono completo
    gdf["area_ha"] = gdf.geometry.area / 1e4
    gdf = gdf[gdf.area_ha >= mmu_px * PX_HA].reset_index(drop=True)
    def d2s(d):
        d = int(d)
        return (datetime(2024, 1, 1) + timedelta(days=d - 1)).strftime("%Y-%m-%d") if 1 <= d <= 366 else None
    dmin = [int(doy_min[l - 1]) if doy_min[l - 1] <= 366 else 0 for l in gdf["lab"]]
    dmax = [int(doy_max[l - 1]) if 1 <= doy_max[l - 1] <= 366 else 0 for l in gdf["lab"]]
    gdf["data_inicial"] = [d2s(x) for x in dmin]
    gdf["data_final"]   = [d2s(x) for x in dmax]
    gdf["doy_inicial"]  = dmin
    gdf["doy_final"]    = dmax
    gdf["duracao_dias"] = [ (b - a + 1) if (a and b and b >= a) else 0 for a, b in zip(dmin, dmax) ]
    gdf["mes_inicial"]  = [int(x.split("-")[1]) if x else 0 for x in gdf["data_inicial"]]
    gdf[["area_ha", "data_inicial", "data_final", "doy_inicial", "doy_final", "duracao_dias", "mes_inicial", "geometry"]].to_file(path, driver="GPKG")
    log(f"GPKG com data (ini/fim): {path} | {len(gdf):,} polígonos")

if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "biatt_unet")
