#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guato_ba · geo_data_science.py
Camada de geociência de dados do estudo de mapeamento automático de queimadas do Pantanal.

GUARDRAIL #1: tudo SEMPRE dentro do bioma Pantanal (pantanal_mgrs_v2.gpkg). Nada fora.

Contém: grid EPSG:5880/20m, máscara do bioma, env PROJ, índices espectrais, validade (SCL+WIW),
carregamento das máscaras de referência FIXAS (nunca regera), downloads BDC/DETER/MapBiomas/HLS,
fusão de revisitas, construção do dataset de patches (entrada já sem água/nuvem), features por pixel
para ML, inferência em janela deslizante (entrada mascarada) e acumulado 2024 TILE-A-TILE.
"""
import os, time, math, glob, json, warnings
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

# --- PROJ/GDAL do env geospatial (NÃO usar o pyproj dinâmico que pega o env base errado) ---
def env_proj():
    os.environ["PROJ_DATA"] = "/opt/conda/envs/geospatial/share/proj"
    os.environ["PROJ_LIB"]  = "/opt/conda/envs/geospatial/share/proj"
    os.environ["GDAL_DATA"] = "/opt/conda/envs/geospatial/share/gdal"
    os.environ["GDAL_NUM_THREADS"] = "1"   # evita threading interno do GDAL (conflito com nosso ThreadPool)
env_proj()

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize, sieve as rio_sieve, shapes as rio_shapes
from rasterio.warp import reproject, Resampling, transform_bounds

# ===================== CONSTANTES =====================
RAIZ       = "/home/jovyan/ID-QUEIMADAS/jupyter"
DADOS      = f"{RAIZ}/bdc_outputs/2024/dados"
GUATO      = f"{RAIZ}/guato_ba"
BIOMA_GPKG = "/home/jovyan/ID-QUEIMADAS/pantanal_mgrs_v2.gpkg"
CUBO_REF   = f"{DADOS}/cubo_20240902.nc"

DST        = CRS.from_epsg(5880)
RES        = 20.0
ESC        = 10000.0                 # DN = 10000 * reflectância
SCL_KEEP   = (4, 5, 7)               # vegetação, solo, cirrus-fino aceitos
WIW_NIR    = 0.1804                  # água (Lefebvre 2019): refl(B8A) < 0.1804
WIW_SW1    = 0.1131                  #                       e refl(B11) < 0.1131
IGN        = 255
PX_HA      = (RES * RES) / 1e4       # 0.04 ha por pixel 20m
BDC_URL    = "https://data.inpe.br/bdc/stac/v1/"
COLLECTION = ["S2_L2A-1"]
# bandas do cubo_corrigido: 1=red,2=b8a,3=b11,4=b12,5=scl
BAND       = {"red": 1, "b8a": 2, "b11": 3, "b12": 4, "scl": 5}

# Máscaras de referência FIXAS (geradas/curadas pelo usuário — NUNCA regerar; ver memória)
REF_2024_1 = f"{RAIZ}/bdc_outputs/2024/mascaras_datas/mascara_datas_1_1_2_2.gpkg"   # 30/05->19/07
REF_2024_2 = f"{RAIZ}/bdc_outputs/2024/mascaras_datas/mascara_datas_2_1_2_2.gpkg"   # 19/07->02/09
REF_2024_3 = f"{RAIZ}/bdc_outputs/2024/mascaras_datas/mascara_datas_3_1_2_2.gpkg"     # 03/08->02/09 (3º par, curado 16-out: nuvem removida)
REF_2024 = [REF_2024_1, REF_2024_2, REF_2024_3]   # 3 pares (datas_3 nova curada, 16-out-2026)
REF_2025 = [f"{RAIZ}/bdc_outputs/2025/out2025_4tiles/mascara_datas_1_1_2_2.gpkg"]   # reeditada 15-out

# Referências secundárias
HLS_DOY  = f"{RAIZ}/acumulado_anual_2024/eventos_dini_dfim.tif"   # banda1 = DOY inicial do sub-evento
HLS_MES  = f"{GUATO}/dados/referencia_hls_mes.tif"               # HLS 2024 quantizado em MÊS (0=sem queimada,1-12)
DETER    = f"{RAIZ}/DETER_NF_PANTANAL.gpkg"
MAPB     = f"{RAIZ}/estudo_pantanal_2024/mascaras/mapbiomas_sigras_2000_policonica.tif"

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ===================== GRID + BIOMA (guardrail) =====================
_GRID = None
def grid_5880():
    """(GH, GW, GGT) do grid global 5880/20m, a partir de cubo_20240902.nc."""
    global _GRID
    if _GRID is not None:
        return _GRID
    import xarray as xr
    ref = xr.open_dataset(CUBO_REF); X = ref.x.values.astype(float); Y = ref.y.values.astype(float); ref.close()
    GX0 = float(X[0] - RES / 2); GYt = float(Y[0] + RES / 2)
    GW = len(X); GH = len(Y); GGT = Affine(RES, 0, GX0, 0, -RES, GYt)
    _GRID = (GH, GW, GGT)
    return _GRID

def _bioma_geom():
    import geopandas as gpd
    from shapely.ops import unary_union
    pg = gpd.read_file(BIOMA_GPKG).to_crs(DST)
    return pg, unary_union(pg.geometry.values)

_BIOMA = None
def mascara_bioma():
    """Matriz booleana (GH,GW) True dentro do Pantanal. GUARDRAIL — aplicar em tudo."""
    global _BIOMA
    if _BIOMA is not None:
        return _BIOMA
    GH, GW, GGT = grid_5880()
    _, geom = _bioma_geom()
    _BIOMA = rasterize([(geom, 1)], out_shape=(GH, GW), transform=GGT, fill=0, dtype="uint8").astype(bool)
    return _BIOMA

def tiles_pantanal():
    """Lista dos tiles MGRS do Pantanal (v2, já recortados no bioma). 51MZR cai por 0 cenas."""
    pg, _ = _bioma_geom()
    return sorted(str(t) for t in pg["tile"].unique())

def geom_por_tile():
    pg, _ = _bioma_geom()
    return {str(r.tile): r.geometry for r in pg.itertuples()}

# ===================== ÍNDICES + VALIDADE =====================
def refl(dn):
    return dn.astype(np.float32) / ESC

def keep_valido(b8a_dn, b11_dn, b12_dn, scl):
    """Pixel válido = bandas>0 AND SCL∈{4,5,7} AND NÃO água(WIW). (entrada já sem água/nuvem)"""
    val  = (b8a_dn > 0) & (b11_dn > 0) & (b12_dn > 0)
    agua = (refl(b8a_dn) < WIW_NIR) & (refl(b11_dn) < WIW_SW1)
    return val & np.isin(scl, SCL_KEEP) & (~agua)

def indices(n, s1, s2):
    """NBR, NBRSWIR, MIRBI a partir de reflectâncias (n=B8A, s1=B11, s2=B12)."""
    nbr = (n - s2) / (n + s2 + 1e-6)
    nsw = (s1 - s2 - 0.02) / (s1 + s2 + 0.1)
    mir = 10.0 * s2 - 9.8 * s1 + 2.0
    return nbr.astype(np.float32), nsw.astype(np.float32), mir.astype(np.float32)

# ===================== REFERÊNCIA FIXA (nunca regera) =====================
def carregar_referencia(ano=2024, clip_bioma=True):
    """Rasteriza as máscaras FIXAS do usuário no grid 5880 e retorna (ref ∩ bioma) booleano.
    NÃO gera nem edita nada — apenas lê os arquivos definitivos."""
    import geopandas as gpd
    GH, GW, GGT = grid_5880()
    arqs = REF_2024 if int(ano) == 2024 else REF_2025
    acc = np.zeros((GH, GW), bool)
    for a in arqs:
        if not os.path.exists(a):
            raise FileNotFoundError(f"referência fixa ausente: {a}")
        g = gpd.read_file(a).to_crs(DST)
        geoms = [(gm, 1) for gm in g.geometry if gm is not None and not gm.is_empty]
        r = rasterize(geoms, out_shape=(GH, GW), transform=GGT, fill=0, dtype="uint8").astype(bool)
        acc |= r
        log(f"  ref {os.path.basename(a)}: {r.sum()*PX_HA:,.0f} ha")
    if clip_bioma:
        acc &= mascara_bioma()
    return acc

def carregar_mapbiomas():
    GH, GW, GGT = grid_5880()
    o = np.zeros((GH, GW), np.uint8)
    with rasterio.open(MAPB) as s:
        reproject(rasterio.band(s, 1), o, src_transform=s.transform, src_crs=DST,
                  dst_transform=GGT, dst_crs=DST, resampling=Resampling.nearest)
    return (o > 0) & mascara_bioma()

def carregar_deter(ano=2024):
    import geopandas as gpd, pandas as pd
    GH, GW, GGT = grid_5880()
    dt = gpd.read_file(DETER); dt["VIEW_DATE"] = pd.to_datetime(dt["VIEW_DATE"], errors="coerce")
    dt = dt[(dt["CLASS_NAME"] == "cicatriz de queimada") & (dt["VIEW_DATE"].dt.year == int(ano))].to_crs(DST)
    geoms = [(g, 1) for g in dt.geometry if g is not None and not g.is_empty]
    r = rasterize(geoms, out_shape=(GH, GW), transform=GGT, fill=0, dtype="uint8").astype(bool)
    return r & mascara_bioma()

def carregar_hls_doy():
    """DOY por pixel (banda1 de eventos_dini_dfim.tif). 0 = sem queimada."""
    with rasterio.open(HLS_DOY) as s:
        return s.read(1)

def carregar_hls_mes(clip_bioma=True):
    """HLS 2024 quantizado em MÊS, reprojetado ao grid 5880/20m do estudo.
    Retorna matriz uint8 (0=sem queimada, 1-12=mês de queimada) ∩ bioma."""
    GH, GW, GGT = grid_5880()
    o = np.zeros((GH, GW), np.uint8)
    with rasterio.open(HLS_MES) as s:
        reproject(rasterio.band(s, 1), o, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=GGT, dst_crs=DST, resampling=Resampling.nearest)
    if clip_bioma:
        o[~mascara_bioma()] = 0
    return o

# ===================== STAC / DOWNLOAD BDC =====================
def _slot_do_tile(geom):
    """Sub-janela (r0,c0,sh,sw,transform) do grid global que cobre a bbox do tile."""
    GH, GW, GGT = grid_5880()
    GX0, GYt = GGT.c, GGT.f
    minx, miny, maxx, maxy = geom.bounds
    c0 = max(0, int(math.floor((minx - GX0) / RES))); r0 = max(0, int(math.floor((GYt - maxy) / RES)))
    c1 = min(GW, int(math.ceil((maxx - GX0) / RES)));  r1 = min(GH, int(math.ceil((GYt - miny) / RES)))
    st = Affine(RES, 0, GX0 + c0 * RES, 0, -RES, GYt - r0 * RES)
    return r0, c0, r1 - r0, c1 - c0, st

def buscar_itens(tile, ini, fim, cloud):
    import pystac_client
    cat = pystac_client.Client.open(BDC_URL)
    search = cat.search(collections=COLLECTION,
                        query={"bdc:tiles": {"in": [tile]}, "eo:cloud_cover": {"lt": cloud}},
                        datetime=f"{ini}T00:00:00Z/{fim}T23:59:59Z")
    return sorted(list(search.get_items()), key=lambda it: it.properties["datetime"])

def _reproj_banda(href, st, sh, sw, resamp=Resampling.nearest):
    o = np.zeros((sh, sw), np.float32 if resamp != Resampling.nearest else np.uint16)
    with rasterio.open(href) as s:
        reproject(rasterio.band(s, 1), o, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=st, dst_crs=DST, resampling=resamp)
    return o

def compor_data(itens_do_dia, st, sh, sw):
    """Une revisitas do MESMO dia: reflectância = média das sobreposições válidas; validade = união.
    Leitura SERIAL (warp do GDAL não é thread-safe aqui — threading segfalta). Retorna (R[4], valido)."""
    sumR = np.zeros((4, sh, sw), np.float32); cnt = np.zeros((sh, sw), np.float32); cval = np.zeros((sh, sw), bool)
    for it in itens_do_dia:
        try:
            red = _reproj_banda(it.assets["B04"].href, st, sh, sw)
            n   = _reproj_banda(it.assets["B8A"].href, st, sh, sw)
            s1  = _reproj_banda(it.assets["B11"].href, st, sh, sw)
            s2  = _reproj_banda(it.assets["B12"].href, st, sh, sw)
            sc  = _reproj_banda(it.assets["SCL"].href, st, sh, sw)
        except Exception as e:
            log(f"    [cena ignorada] {e}"); continue
        vr = keep_valido(n, s1, s2, sc)
        if not vr.any(): continue
        R = np.stack([refl(red), refl(n), refl(s1), refl(s2)], 0)
        sumR[:, vr] += R[:, vr]; cnt[vr] += 1; cval |= vr
    return (sumR / np.maximum(cnt, 1)).astype(np.float32), cval

# ===================== MÉTRICAS (artigo) =====================
def metricas_artigo(P, A, dom=None):
    if dom is None:
        dom = np.ones_like(P, bool)
    P = P & dom; A = A & dom
    a = float((P & A).sum()); b = float((P & ~A).sum()); c = float((~P & A).sum()); N = float(dom.sum())
    d = N - a - b - c; ac = a + c; pp = a + b; ha = lambda x: x * PX_HA
    # CE (comissão) = FP/(TP+FP) = 1 - precisão (PADRÃO). CE_ref = FP/(TP+FN) (antiga, FP/referência).
    return dict(F1=2*a/(2*a+b+c) if (2*a+b+c) else 0.0, IoU=a/(a+b+c) if (a+b+c) else 0.0,
                OA=(a+d)/N if N else 0.0, OE=c/ac if ac else 0.0, CE=b/pp if pp else 0.0, CE_ref=b/ac if ac else 0.0,
                BIAS=(a+b)/ac if ac else 0.0, recall=a/ac if ac else 0.0,
                precisao=a/pp if pp else 0.0,
                area_modelo_ha=ha(a+b), area_ref_ha=ha(ac), acerto_ha=ha(a), FP_ha=ha(b), FN_ha=ha(c))

def metricas_pixel(TP, FP, FN, TN):
    # CE (comissão) = FP/(TP+FP) = 1 - precisão  (convenção PADRÃO: erro relativo à área PREDITA).
    # OE (omissão)  = FN/(TP+FN) = 1 - recall    (erro relativo à área de REFERÊNCIA).
    # CE_ref = FP/(TP+FN) é a convenção ANTIGA (FP relativo à referência); mantida p/ compatibilidade.
    N = TP + FP + FN + TN; ac = TP + FN; pp = TP + FP
    return dict(F1=2*TP/(2*TP+FP+FN) if (2*TP+FP+FN) else 0.0, IoU=TP/(TP+FP+FN) if (TP+FP+FN) else 0.0,
                OE=FN/ac if ac else 0.0, CE=FP/pp if pp else 0.0, CE_ref=FP/ac if ac else 0.0,
                recall=TP/ac if ac else 0.0, precisao=TP/pp if pp else 0.0, OA=(TP+TN)/N if N else 0.0)

# ===================== SAÍDA RASTER =====================
def salvar_raster(path, arr, nodata=0, dtype="uint8", overviews=True):
    GH, GW, GGT = grid_5880()
    prof = dict(driver="GTiff", height=GH, width=GW, count=1, dtype=dtype, crs=DST, transform=GGT,
                nodata=nodata, compress="DEFLATE", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype(dtype), 1)
    if overviews:
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", COMPRESS_OVERVIEW="DEFLATE"):
            with rasterio.open(path, "r+") as d:
                d.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)

def filtrar_min_px(mask_bool, min_px=12):
    """Remove componentes < min_px via sieve conn-8 (fix de vetorização)."""
    return rio_sieve(mask_bool.astype(np.uint8), size=min_px, connectivity=8).astype(bool)

# ===================== DATASET DE PATCHES (SCL_MIN=0.30, guardrail embutido) =====================
PATCH = 256; SCL_MIN = 0.30; BLOCK = 2560   # patch 256 (mais contexto de vizinhança)
CUBOS_2024 = {"0530": f"{DADOS}/cubo_20240530_corrigido.tif",
              "0719": f"{DADOS}/cubo_20240719_corrigido.tif",
              "0803": f"{DADOS}/cubo_20240803_corrigido.tif",
              "0902": f"{DADOS}/cubo_20240902_corrigido.tif"}
PARES_2024 = [("1", "0530", "0719", REF_2024_1),   # 30/05->19/07
              ("2", "0719", "0902", REF_2024_2),   # 19/07->02/09
              ("3", "0803", "0902", REF_2024_3)]   # 03/08->02/09 (3º par, datas_3 curada 16-out)
MOS_2025 = {"t1": f"{RAIZ}/bdc_outputs/2025/out2025_4tiles/mosaico_4tiles_20250904_comum.tif",
            "t2": f"{RAIZ}/bdc_outputs/2025/out2025_4tiles/mosaico_4tiles_20251031_comum.tif"}
DATA_DIR = f"{GUATO}/data"

def _ler_cubo(path):
    """Lê [red,b8a,b11,b12,scl] (uint16) do cubo. Retorna dict de bandas + (transform,H,W)."""
    with rasterio.open(path) as s:
        GT = s.transform; H, W = s.shape
        d = {k: s.read(BAND[k]) for k in ("red", "b8a", "b11", "b12", "scl")}
    return d, GT, H, W

def _rasteriza_ref(arq, GT, H, W):
    import geopandas as gpd
    g = gpd.read_file(arq).to_crs(DST)
    geoms = [(gm, 1) for gm in g.geometry if gm is not None and not gm.is_empty]
    return rasterize(geoms, out_shape=(H, W), transform=GT, fill=0, dtype="uint8").astype(bool)

def _bioma_no_grid(GT, H, W):
    _, geom = _bioma_geom()
    return rasterize([(geom, 1)], out_shape=(H, W), transform=GT, fill=0, dtype="uint8").astype(bool)

def _tiles_de_par(d1, d2, ref, biom, nome):
    """Tila step=PATCH; mantém patch se (válido ∩ bioma).mean() >= SCL_MIN. Entrada = DN cru;
    gabarito: burn=1 onde ref∩válido∩bioma, 0 onde válido∩bioma∩~ref, IGN fora disso (inclui fora do bioma e nuvem/água)."""
    H, W = d1["scl"].shape
    val = keep_valido(d1["b8a"], d1["b11"], d1["b12"], d1["scl"]) & \
          keep_valido(d2["b8a"], d2["b11"], d2["b12"], d2["scl"])
    valb = val & biom
    T1, T2, Y, G = [], [], [], []
    for r0 in range(0, H - PATCH + 1, PATCH):
        for c0 in range(0, W - PATCH + 1, PATCH):
            sl = (slice(r0, r0+PATCH), slice(c0, c0+PATCH))
            vp = valb[sl]
            if vp.mean() < SCL_MIN:
                continue
            t1 = np.stack([d1["red"][sl], d1["b8a"][sl], d1["b11"][sl], d1["b12"][sl]], 0).astype(np.uint16)
            t2 = np.stack([d2["red"][sl], d2["b8a"][sl], d2["b11"][sl], d2["b12"][sl]], 0).astype(np.uint16)
            mk = np.where(ref[sl] & vp, 1, 0).astype(np.uint8); mk[~vp] = IGN
            T1.append(t1); T2.append(t2); Y.append(mk)
            G.append(f"{r0//BLOCK}_{c0//BLOCK}")
    log(f"  par {nome}: {len(Y)} patches (válido∩bioma>= {SCL_MIN:.0%})")
    return T1, T2, Y, G

def construir_dataset_2024(forcar=False, pares=None, prefixo="train"):
    """Reconstrói o dataset de treino 2024 → .npy em data/ com prefixo <prefixo>. SCL_MIN=0.30, guardrail.
    pares=subconjunto de PARES_2024 (default todos). Permite variantes: train_v3, train_v12, train_v123..."""
    os.makedirs(DATA_DIR, exist_ok=True)
    pares = pares if pares is not None else PARES_2024
    out = {k: f"{DATA_DIR}/{prefixo}_{k}.npy" for k in ("T1", "T2", "Y", "G")}
    if (not forcar) and all(os.path.exists(v) for v in out.values()):
        log(f"dataset {prefixo} já existe (cache)."); return out
    GH, GW, GGT = grid_5880(); biom = mascara_bioma()
    T1a, T2a, Ya, Ga = [], [], [], []
    for nome, k1, k2, ref_arq in pares:
        log(f"[{prefixo}] lendo par datas_{nome} ({k1}->{k2})…")
        d1, GT, H, W = _ler_cubo(CUBOS_2024[k1]); d2, _, _, _ = _ler_cubo(CUBOS_2024[k2])
        ref = _rasteriza_ref(ref_arq, GGT, GH, GW)
        t1, t2, y, g = _tiles_de_par(d1, d2, ref, biom, f"datas_{nome}")
        T1a += t1; T2a += t2; Ya += y; Ga += g
        del d1, d2, ref
    T1 = np.stack(T1a); T2 = np.stack(T2a); Y = np.stack(Ya); G = np.array(Ga)
    np.save(out["T1"], T1); np.save(out["T2"], T2); np.save(out["Y"], Y); np.save(out["G"], G)
    log(f"✅ dataset {prefixo}: {len(Y)} patches → {DATA_DIR} (T1 {T1.shape})")
    return out

def construir_teste_2025(forcar=False):
    """Teste OOT 2025 (par 0904->1031) recortado ao Pantanal → .npy. SCL_MIN=0.30, guardrail."""
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {k: f"{DATA_DIR}/test2025_{k}.npy" for k in ("T1", "T2", "Y", "G")}
    if (not forcar) and all(os.path.exists(v) for v in out.values()):
        log("teste 2025 já existe (cache)."); return out
    d1, GT, H, W = _ler_cubo(MOS_2025["t1"]); d2, _, _, _ = _ler_cubo(MOS_2025["t2"])
    ref = _rasteriza_ref(REF_2025[0], GT, H, W); biom = _bioma_no_grid(GT, H, W)
    t1, t2, y, g = _tiles_de_par(d1, d2, ref, biom, "teste2025")
    T1 = np.stack(t1); T2 = np.stack(t2); Y = np.stack(y); G = np.array(g)
    np.save(out["T1"], T1); np.save(out["T2"], T2); np.save(out["Y"], Y); np.save(out["G"], G)
    log(f"✅ teste 2025: {len(Y)} patches → {DATA_DIR}")
    return out

def carregar_npy(split):
    pre = split   # prefixo do cache: "train", "test2025", ou variantes "train_v3"/"train_v12"/...
    return (np.load(f"{DATA_DIR}/{pre}_T1.npy"), np.load(f"{DATA_DIR}/{pre}_T2.npy"),
            np.load(f"{DATA_DIR}/{pre}_Y.npy"), np.load(f"{DATA_DIR}/{pre}_G.npy", allow_pickle=True))

# ===================== FEATURES POR PIXEL (ML) =====================
def _feats_chunk(T1sl, T2sl):
    r1 = T1sl.astype(np.float32) / ESC; r2 = T2sl.astype(np.float32) / ESC
    nbr1, nsw1, mir1 = indices(r1[:, 1], r1[:, 2], r1[:, 3])
    nbr2, nsw2, mir2 = indices(r2[:, 1], r2[:, 2], r2[:, 3])
    feats = np.concatenate([r1, r2, (nbr1-nbr2)[:, None], (nsw1-nsw2)[:, None], (mir1-mir2)[:, None]], axis=1)
    return feats.transpose(0, 2, 3, 1).reshape(-1, feats.shape[1])

def extrair_features_pixel(T1, T2, Y, n_por_classe=250000, seed=42, chunk=800, balancear=True):
    """Features por pixel VÁLIDO (Y!=IGN), EM CHUNKS. Features(11): t1[4],t2[4],dNBR,dNBRSWIR,dMIRBI (delta=t1-t2).
    balancear=True → amostra igual por classe (treino/HPO); balancear=False → proporção NATURAL (avaliação realista)."""
    rng = np.random.default_rng(seed)
    if balancear:
        posX, negX = [], []
        for st in range(0, len(T1), chunk):
            sl = slice(st, st + chunk)
            X = _feats_chunk(T1[sl], T2[sl]); yv = Y[sl].reshape(-1)
            v = yv != IGN; X = X[v]; yv = yv[v]
            p = X[yv == 1]; n = X[yv == 0]
            if len(p): posX.append(p.astype(np.float32))
            if len(n):
                k = rng.choice(len(n), min(len(n), n_por_classe), replace=False); negX.append(n[k].astype(np.float32))
        pos = np.concatenate(posX) if posX else np.zeros((0, 11), np.float32)
        neg = np.concatenate(negX) if negX else np.zeros((0, 11), np.float32)
        if len(pos) > n_por_classe: pos = pos[rng.choice(len(pos), n_por_classe, replace=False)]
        if len(neg) > n_por_classe: neg = neg[rng.choice(len(neg), n_por_classe, replace=False)]
        X = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos), np.uint8), np.zeros(len(neg), np.uint8)])
    else:
        # proporção natural, com teto global (preserva o desbalanceamento real)
        cap = n_por_classe * 8; Xl, yl = [], []
        for st in range(0, len(T1), chunk):
            sl = slice(st, st + chunk)
            X = _feats_chunk(T1[sl], T2[sl]); yv = Y[sl].reshape(-1)
            v = yv != IGN; Xl.append(X[v].astype(np.float32)); yl.append(yv[v].astype(np.uint8))
        X = np.concatenate(Xl); y = np.concatenate(yl)
        if len(X) > cap:
            k = rng.choice(len(X), cap, replace=False); X, y = X[k], y[k]
    idx = rng.permutation(len(X)); return X[idx], y[idx]

# ===================== INFERÊNCIA (entrada mascarada) + ACUMULADO TILE-A-TILE =====================
def _coords(L, patch=PATCH, ov=16):
    st = patch - ov; c = list(range(0, max(1, L - patch + 1), st))
    if c and c[-1] != L - patch and L > patch: c.append(L - patch)
    return c

if __name__ == "__main__":
    GH, GW, GGT = grid_5880(); B = mascara_bioma()
    log(f"grid {GH}x{GW} | bioma = {B.sum()*PX_HA:,.0f} ha | tiles = {len(tiles_pantanal())}")
    ref = carregar_referencia(2024); fora = int((ref & ~B).sum())
    log(f"referência 2024 ∩ bioma = {ref.sum()*PX_HA:,.0f} ha | pixels fora do bioma = {fora}  (deve ser 0)")
