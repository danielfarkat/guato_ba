#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guato_ba · accumulate_landsat.py — ACUMULADO 2024 LANDSAT (BDC 'landsat-2', C2L2), TILE-A-TILE.

Igual ao Sentinel (accumulate_tile), mas:
  - coleção 'landsat-2' (Landsat 8/9 C2 L2), busca por bbox do tile MGRS (grid 20m do estudo);
  - 3 bandas: NIR=B5(nir08), SWIR1=B6(swir16), SWIR2=B7(swir22) -> equivalem a b8a,b11,b12 do S2;
  - reflectância C2L2 = DN*2.75e-5 - 0.2 (com OFFSET; ratio precisa da conversão correta), clip [0,1];
  - validade: QA_PIXEL ∈ {1,322,386,834,898,1346} (equivale ao SCL) & bandas>0 & ~WIW(água);
  - modelo = biatt_unet (3 bandas, sem red) TRANSFERIDO do Sentinel (bandas [1,2,3]).
Processa tile a tile no MESMO grid 20m/bioma do Sentinel (comparável), com checkpoint .npz por tile
e dissolve final (GPKG com data_inicial/data_final). GUARDRAIL: tudo dentro do Pantanal.
"""
import os, time, json, warnings
warnings.filterwarnings("ignore")
# PROJ/GDAL do env geospatial ANTES de importar rasterio/torch (script rodado direto, não via main.py)
for _k in ("PROJ_DATA", "PROJ_LIB"): os.environ[_k] = "/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"] = "/opt/conda/envs/geospatial/share/gdal"
import numpy as np, torch
from concurrent.futures import ProcessPoolExecutor
from rasterio.features import rasterize
import geo_data_science as G
from geo_data_science import (log, grid_5880, mascara_bioma, geom_por_tile, _slot_do_tile,
                              DST, PX_HA, salvar_raster, filtrar_min_px, BDC_URL)
from redes import construir_dl, MODELOS_DL
import accumulate_tile as AT

INI = "2024-01-01"; FIM = "2024-12-31"; CLOUD = 10.0; IN_BIOMA_MIN = 0.50
THR = 0.90; MIN_PX = 24; NPROC = 5
QA_KEEP = (1, 322, 386, 834, 898, 1346)   # equivalente ao SCL (Landsat C2 QA_PIXEL)
L_SCALE = 2.75e-5; L_OFFSET = -0.2         # reflectância C2L2 = DN*scale + offset
WIW_NIR = 0.1804; WIW_SW1 = 0.1131
COLLECTION_L = "landsat-2"
ASSETS_L = {"nir": "nir08", "swir1": "swir16", "swir2": "swir22", "qa": "qa_pixel"}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACU_DIR = AT.ACU_DIR

# ---------- busca STAC Landsat por bbox do tile ----------
def buscar_itens_landsat(geom5880, ini, fim, cloud):
    import pystac_client
    from rasterio.warp import transform_bounds
    minx, miny, maxx, maxy = geom5880.bounds
    w, s, e, n = transform_bounds(DST, "EPSG:4326", minx, miny, maxx, maxy)
    cat = pystac_client.Client.open(BDC_URL)
    search = cat.search(collections=[COLLECTION_L], bbox=[w, s, e, n],
                        query={"eo:cloud_cover": {"lt": cloud}},
                        datetime=f"{ini}T00:00:00Z/{fim}T23:59:59Z")
    return sorted(list(search.get_items()), key=lambda it: it.properties["datetime"])

# ---------- worker (processo): compõe UMA data Landsat no slot do tile ----------
def _compose_worker_landsat(payload):
    import os as _os
    _os.environ["PROJ_DATA"] = "/opt/conda/envs/geospatial/share/proj"
    _os.environ["PROJ_LIB"] = "/opt/conda/envs/geospatial/share/proj"
    _os.environ["GDAL_DATA"] = "/opt/conda/envs/geospatial/share/gdal"
    import numpy as _np, rasterio
    from rasterio import Affine as _Aff
    from rasterio.crs import CRS as _CRS
    from rasterio.warp import reproject as _rp, Resampling as _Rs
    scenes, st_t, sh, sw, qa_keep = payload
    st = _Aff(*st_t); dst = _CRS.from_epsg(5880); QA = tuple(qa_keep)
    def rb(href, dtype):
        o = _np.zeros((sh, sw), dtype)
        with rasterio.open(href) as s:
            _rp(rasterio.band(s, 1), o, src_transform=s.transform, src_crs=s.crs,
                dst_transform=st, dst_crs=dst, resampling=_Rs.nearest)
        return o
    # saída layout [red(zeros), nir, swir1, swir2] em reflectância (bandas [1,2,3] usadas pelo modelo)
    sumR = _np.zeros((4, sh, sw), _np.float32); cnt = _np.zeros((sh, sw), _np.float32); cval = _np.zeros((sh, sw), bool)
    for sc in scenes:
        try:
            nir = rb(sc["nir"], _np.uint16); s1 = rb(sc["swir1"], _np.uint16)
            s2 = rb(sc["swir2"], _np.uint16); qa = rb(sc["qa"], _np.uint16)
        except Exception:
            continue
        rn = _np.clip(nir.astype(_np.float32) * L_SCALE + L_OFFSET, 0, 1)
        r1 = _np.clip(s1.astype(_np.float32) * L_SCALE + L_OFFSET, 0, 1)
        r2 = _np.clip(s2.astype(_np.float32) * L_SCALE + L_OFFSET, 0, 1)
        agua = (rn < WIW_NIR) & (r1 < WIW_SW1)
        # QA_PIXEL C2L2 por BITS (equivale ao SCL): clear = sem fill(0)/dilatada(1)/cirrus(2)/nuvem(3)/sombra(4)
        qa_clear = (qa & 0b11111) == 0
        val = (nir > 0) & (s1 > 0) & (s2 > 0) & qa_clear & (~agua)
        if not val.any():
            continue
        sumR[1, val] += rn[val]; sumR[2, val] += r1[val]; sumR[3, val] += r2[val]
        cnt[val] += 1; cval |= val
    c = _np.maximum(cnt, 1)
    sumR[1] /= c; sumR[2] /= c; sumR[3] /= c
    return sumR.astype(_np.float32), cval

# ---------- processa um tile (mesma lógica do Sentinel, com compose Landsat) ----------
def _processar_tile_landsat(model, bandas, mean, std, tile, geom, cloud, in_bioma_min, thr, qa_keep, pool):
    r0, c0, sh, sw, st = _slot_do_tile(geom)
    if sh <= 0 or sw <= 0: return None
    tile_mask = rasterize([(geom, 1)], out_shape=(sh, sw), transform=st, fill=0, dtype="uint8").astype(bool)
    area_tile = int(tile_mask.sum())
    if area_tile == 0: return None
    try:
        itens = buscar_itens_landsat(geom, INI, FIM, cloud)
    except Exception as e:
        log(f"[{tile}] STAC Landsat erro: {e}"); return None
    pordia = {}
    for it in itens:
        dia = it.properties["datetime"][:10]; pordia.setdefault(dia, []).append(it)
    st_t = (st.a, st.b, st.c, st.d, st.e, st.f)
    dias = sorted(pordia)
    payloads = [([{k: it.assets[a].href for k, a in ASSETS_L.items() if a in it.assets}
                  for it in pordia[dia]], st_t, sh, sw, qa_keep) for dia in dias]
    comp = {}; ncenas = 0
    for dia, (Rr, val) in zip(dias, pool.map(_compose_worker_landsat, payloads)):
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
        prob = AT._infere_par(model, bandas, mean, std, Ra, Rb, dom, sh, sw)
        det = (prob >= thr) & dom
        novo = det & (det_tile == 0); det_tile[novo] = AT._doy(ds[k+1])
        burned_tile |= det
        prob_tile = np.maximum(prob_tile, np.where(dom, prob, 0.0))
        anal_tile |= dom
    return dict(r0=r0, c0=c0, sh=sh, sw=sw, tile_mask=tile_mask, area_tile=area_tile,
                burned=(burned_tile & tile_mask), anal=(anal_tile & tile_mask),
                prob=((prob_tile * 255).astype(np.uint8) * tile_mask),
                det=np.where(tile_mask, det_tile, 0).astype(np.uint16),
                ncenas=ncenas, ndatas=len(ds))

def listar_tiles_landsat_pantanal():
    """Lista os WRS path/row do Landsat que intersectam o Pantanal (apenas informativo)."""
    import pystac_client
    from rasterio.warp import transform_bounds
    _, geom = G._bioma_geom()
    w, s, e, n = transform_bounds(DST, "EPSG:4326", *geom.bounds)
    cat = pystac_client.Client.open(BDC_URL)
    search = cat.search(collections=[COLLECTION_L], bbox=[w, s, e, n],
                        datetime="2024-08-01T00:00:00Z/2024-08-31T23:59:59Z")
    wrs = sorted({t for it in search.get_items() for t in it.properties.get("bdc:tiles", [])})
    log(f"Landsat WRS path/row no Pantanal (ago/2024): {len(wrs)} -> {wrs}")
    return wrs

def run_pantanal_landsat(modelo="biatt_unet", cloud=CLOUD, in_bioma_min=IN_BIOMA_MIN, thr=THR,
                         mmu_px=MIN_PX, qa_keep=QA_KEEP, tag="_landsat_pantanal"):
    """Acumulado Landsat do Pantanal inteiro, tile a tile (grid 20m) com checkpoint + dissolve final.
    Usa a biatt_unet (3 bandas, sem red) treinada no Sentinel (transfer)."""
    assert modelo in MODELOS_DL
    ckp = f"{G.GUATO}/deeplearning/{modelo}/{modelo}_best.pkl"
    ck = torch.load(ckp, map_location=DEV, weights_only=False); bandas = ck["bandas"]
    mean = ck["mean"]; std = ck["std"]
    model = construir_dl(modelo, p=0.0, pretrained=False).to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    GH, GW, GGT = grid_5880(); BIOMA = mascara_bioma()
    ACUM = np.zeros((GH, GW), bool); ANALISADO = np.zeros((GH, GW), bool)
    PROB = np.zeros((GH, GW), np.uint8); DET = np.zeros((GH, GW), np.uint16)
    geoms = geom_por_tile(); alltiles = sorted(geoms)
    CKDIR = f"{ACU_DIR}/_ckpt_{modelo}{tag}"; os.makedirs(CKDIR, exist_ok=True)
    log(f"LANDSAT {modelo}{tag} | {len(alltiles)} tiles | cloud<{cloud}% | area>={in_bioma_min:.0%} | THR={thr} | MMU={mmu_px}px | QA=bits_clear(&31==0) | bandas={bandas}")
    pool = ProcessPoolExecutor(max_workers=NPROC); t_all = time.time()
    for ti, tile in enumerate(alltiles):
        ckf = f"{CKDIR}/{tile}.npz"; t0 = time.time()
        if os.path.exists(ckf):
            z = np.load(ckf)
            if int(z["sh"]) == 0: log(f"[{tile}] ckpt vazio ({ti+1}/{len(alltiles)})"); continue
            res = dict(r0=int(z["r0"]), c0=int(z["c0"]), sh=int(z["sh"]), sw=int(z["sw"]),
                       tile_mask=z["tile_mask"], burned=z["burned"], anal=z["anal"], prob=z["prob"], det=z["det"])
            AT._merge_tile(res, ACUM, ANALISADO, PROB, DET)
            log(f"[{tile}] ckpt reaproveitado | queimado {res['burned'].sum()*PX_HA:,.0f}ha ({ti+1}/{len(alltiles)})"); continue
        res = _processar_tile_landsat(model, bandas, mean, std, tile, geoms[tile], cloud, in_bioma_min, thr, qa_keep, pool)
        if res is None:
            np.savez_compressed(ckf, r0=0, c0=0, sh=0, sw=0)
            log(f"[{tile}] vazio/sem dados ({ti+1}/{len(alltiles)})"); continue
        np.savez_compressed(ckf, r0=res["r0"], c0=res["c0"], sh=res["sh"], sw=res["sw"],
                            tile_mask=res["tile_mask"], burned=res["burned"], anal=res["anal"],
                            prob=res["prob"], det=res["det"])
        AT._merge_tile(res, ACUM, ANALISADO, PROB, DET)
        log(f"[{tile}] {res['ncenas']} cenas<{cloud}% | {res['ndatas']} datas(>= {in_bioma_min:.0%}) | queimado {res['burned'].sum()*PX_HA:,.0f}ha | analisado {res['anal'].sum()/max(res['area_tile'],1)*100:.0f}% ({ti+1}/{len(alltiles)}, {time.time()-t0:.0f}s)")
    pool.shutdown(wait=True)
    ACUM &= BIOMA; ANALISADO &= BIOMA
    ACUM = filtrar_min_px(ACUM, mmu_px) & BIOMA; DET[~ACUM] = 0
    salvar_raster(f"{ACU_DIR}/acum2024_{modelo}{tag}.tif", ACUM.astype(np.uint8))
    salvar_raster(f"{ACU_DIR}/analisado_{modelo}{tag}.tif", ANALISADO.astype(np.uint8))
    PROB[~BIOMA] = 0; salvar_raster(f"{ACU_DIR}/prob2024_{modelo}{tag}.tif", PROB, nodata=0, dtype="uint8")
    salvar_raster(f"{ACU_DIR}/detdoy_{modelo}{tag}.tif", DET, nodata=0, dtype="uint16")
    AT._gpkg_com_data(ACUM, DET, GGT, f"{ACU_DIR}/acum2024_{modelo}{tag}.gpkg", mmu_px)
    cob = ANALISADO.sum() / max(BIOMA.sum(), 1) * 100
    resumo = dict(sensor="landsat", modelo=modelo, tag=tag, cloud=cloud, area_limpa_min=in_bioma_min,
                  thr=thr, mmu_px=mmu_px, qa="bits_clear(&31==0)",
                  area_queimada_ha=float(ACUM.sum()*PX_HA), cobertura_pantanal_pct=float(cob),
                  n_tiles=len(alltiles), tempo_min=round((time.time()-t_all)/60, 1))
    json.dump(resumo, open(f"{ACU_DIR}/resumo_{modelo}{tag}.json", "w"), indent=2)
    log(f"✅ LANDSAT {modelo}{tag}: {ACUM.sum()*PX_HA:,.0f} ha | cobertura {cob:.1f}% | {(time.time()-t_all)/60:.0f}min → {ACU_DIR}")
    return resumo

if __name__ == "__main__":
    import sys
    listar_tiles_landsat_pantanal()
    run_pantanal_landsat(sys.argv[1] if len(sys.argv) > 1 else "biatt_unet")
