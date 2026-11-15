#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mergulho no tile 21KVT, 2023 (modelo unet_gsplit_norte, cloud<=10%, area-limpa>=95%, THR 0.50).
Gera: (2) GeoTIFF por data limpa [SWIR2,NIR8A,Red]; (3) GeoTIFF por par consecutivo [dNDVI,NBR_t2,dNBR];
(1) acumulado (raster+vetor); (4) comparacao vs HLS 2023. Saidas: results/acumulado/tile21kvt_2023/."""
import os
os.environ["GUATO_NORM"]="gn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, rasterio, geopandas as gpd, pandas as pd, torch
from concurrent.futures import ProcessPoolExecutor
from rasterio.features import rasterize, shapes
from shapely.geometry import shape
import accumulate_tile as A
from accumulate_tile import (_slot_do_tile, _compose_worker, _infere_par, _doy, log)
import geo_data_science as g
from geo_data_science import (buscar_itens, geom_por_tile, mascara_bioma, grid_5880,
                              metricas_artigo, PX_HA, DST)
from redes import construir_dl

TILE="21KVT"; A.INI="2023-01-01"; A.FIM="2023-12-31"
CLOUD=10.0; IN_BIOMA_MIN=0.95; THR=0.50; MMU=24; SCL_KEEP=(4,5,7)
OUT=f"{g.GUATO}/results/acumulado/tile21kvt_2023"; os.makedirs(OUT,exist_ok=True)
CKPT=f"{g.GUATO}/deeplearning/unet_gsplit_norte/unet_best.pkl"
DEV=A.DEV
log(f"### TILE {TILE} 2023 | cloud<{CLOUD}% area-limpa>={IN_BIOMA_MIN:.0%} THR={THR} MMU={MMU} ###")

# ---- modelo ----
ck=torch.load(CKPT, map_location=DEV, weights_only=False); bandas=ck["bandas"]; mean=ck["mean"]; std=ck["std"]
model=construir_dl("unet", p=0.0, pretrained=False).to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()

# ---- slot + mascara do tile ----
geom=geom_por_tile()[TILE]; r0,c0,sh,sw,st=_slot_do_tile(geom)
tile_mask=rasterize([(geom,1)],out_shape=(sh,sw),transform=st,fill=0,dtype="uint8").astype(bool)
area_tile=int(tile_mask.sum())
GH,GW,GGT=grid_5880(); bioma_tile=mascara_bioma()[r0:r0+sh,c0:c0+sw]
log(f"slot: {sh}x{sw} | area_tile={area_tile*PX_HA:,.0f} ha")

def grava(path,arr,dtype,descr,nodata=None):
    arr=np.ascontiguousarray(arr.astype(dtype))
    prof=dict(driver="GTiff",height=sh,width=sw,count=arr.shape[0],dtype=dtype,crs=DST,transform=st,compress="deflate")
    if nodata is not None: prof["nodata"]=nodata
    with rasterio.open(path,"w",**prof) as d:
        d.write(arr)
        for i,n in enumerate(descr,1): d.set_band_description(i,n)

# ---- datas limpas (espelha _processar_tile) ----
itens=buscar_itens(TILE,A.INI,A.FIM,CLOUD)
pordia={}
for it in itens: pordia.setdefault(it.properties["datetime"][:10],[]).append(it)
st_t=(st.a,st.b,st.c,st.d,st.e,st.f); dias=sorted(pordia)
payloads=[([{b:it.assets[b].href for b in ("B04","B8A","B11","B12","SCL") if b in it.assets} for it in pordia[d]],
           st_t,sh,sw,SCL_KEEP) for d in dias]
comp={}
with ProcessPoolExecutor(max_workers=A.NPROC) as pool:
    for d,(Rr,val) in zip(dias,pool.map(_compose_worker,payloads)):
        vt=val&tile_mask
        if area_tile and vt.sum()/area_tile>=IN_BIOMA_MIN: comp[d]=(Rr,vt)
ds=sorted(comp)
log(f"datas limpas (>= {IN_BIOMA_MIN:.0%}): {len(ds)} -> {ds}")

# ---- (2) GeoTIFF por data: [SWIR2, NIR8A, Red] reflect x10000 ----
for d in ds:
    Rr,vt=comp[d]                                    # Rr = [red, b8a(NIR), b11(SWIR1), b12(SWIR2)]
    comp_rgb=np.stack([Rr[3],Rr[1],Rr[0]],0)*10000.0 # SWIR2, NIR8A, Red
    comp_rgb[:,~vt]=0
    grava(f"{OUT}/data_{d.replace('-','')}_swir2_nir_red.tif", comp_rgb, "uint16",
          ["SWIR2_B12","NIR8A_B8A","Red_B04"], nodata=0)
log(f"[2] {len(ds)} GeoTIFF por data gravados")

# ---- (3) GeoTIFF por par: [dNDVI, NBR_t2, dNBR] ----
def ndvi(R): return (R[1]-R[0])/(R[1]+R[0]+1e-6)
def nbr(R):  return (R[1]-R[3])/(R[1]+R[3]+1e-6)
for k in range(len(ds)-1):
    d1,d2=ds[k],ds[k+1]; (Ra,va),(Rb,vb)=comp[d1],comp[d2]; dom=va&vb&tile_mask
    dndvi=ndvi(Ra)-ndvi(Rb); nbr2=nbr(Rb); dnbr=nbr(Ra)-nbr(Rb)
    diff=np.stack([dndvi,nbr2,dnbr],0).astype(np.float32); diff[:,~dom]=np.nan
    grava(f"{OUT}/par_{d1.replace('-','')}_{d2.replace('-','')}_diff.tif", diff, "float32",
          ["dNDVI","NBR_t2","dNBR"], nodata=float("nan"))
log(f"[3] {max(0,len(ds)-1)} GeoTIFF de diferenca por par gravados")

# ---- (1) acumulado: roda modelo por par, une ----
burned=np.zeros((sh,sw),bool); anal=np.zeros((sh,sw),bool)
prob=np.zeros((sh,sw),np.float32); det=np.zeros((sh,sw),np.uint16)
for k in range(len(ds)-1):
    d1,d2=ds[k],ds[k+1]; (Ra,va),(Rb,vb)=comp[d1],comp[d2]; dom=va&vb&tile_mask
    if not dom.any(): continue
    pr=_infere_par(model,bandas,mean,std,Ra,Rb,dom,sh,sw)
    de=(pr>=THR)&dom
    novo=de&(det==0); det[novo]=_doy(d2)
    burned|=de; prob=np.maximum(prob,np.where(dom,pr,0.0)); anal|=dom
burned=A.filtrar_min_px(burned,MMU)&tile_mask; det[~burned]=0
grava(f"{OUT}/acum2023_21kvt.tif",burned[None].astype(np.uint8),"uint8",["queimada"],nodata=0)
grava(f"{OUT}/analisado2023_21kvt.tif",anal[None].astype(np.uint8),"uint8",["analisado"],nodata=0)
grava(f"{OUT}/prob2023_21kvt.tif",(prob*255)[None].astype(np.uint8),"uint8",["prob_x255"],nodata=0)
# vetor (polígonos + area_ha)
polys=[(shape(gm),v) for gm,v in shapes(burned.astype("uint8"),mask=burned,transform=st) if v==1]
gdf=gpd.GeoDataFrame({"classe":["queimada"]*len(polys)},
                     geometry=[p for p,_ in polys],crs=DST)
gdf["area_ha"]=gdf.geometry.area/1e4
gdf.to_file(f"{OUT}/acum2023_21kvt.gpkg",driver="GPKG")
log(f"[1] acumulado 21KVT 2023 = {burned.sum()*PX_HA:,.0f} ha | analisado {anal.sum()/max(area_tile,1)*100:.1f}% | {len(gdf)} poligonos")

# ---- (4) comparacao vs HLS 2023 ----
HLS="/home/jovyan/ID-QUEIMADAS/jupyter/bdc_outputs/2023/aqm_2023.gpkg"
hg=gpd.read_file(HLS).to_crs(DST)
hls=rasterize([(gm,1) for gm in hg.geometry if gm and not gm.is_empty],out_shape=(sh,sw),transform=st,fill=0,dtype="uint8").astype(bool)
dom_t=tile_mask&bioma_tile
rows=[]
for dn,dom in [("tile",dom_t),("tile_analisado",dom_t&anal)]:
    m=metricas_artigo(burned,hls,dom)
    log(f"[4] vs HLS [{dn:15s}] IoU={m['IoU']:.3f} F1={m['F1']:.3f} OE={m['OE']:.3f} CE={m['CE']:.3f} area_mod={m['area_modelo_ha']:,.0f} area_hls={m['area_ref_ha']:,.0f}")
    rows.append(dict(dominio=dn,IoU=round(m["IoU"],3),F1=round(m["F1"],3),OE=round(m["OE"],3),CE=round(m["CE"],3),
                     area_modelo_ha=round(m["area_modelo_ha"],0),area_hls_ha=round(m["area_ref_ha"],0)))
pd.DataFrame(rows).to_csv(f"{OUT}/comparacao_hls_21kvt.csv",index=False)
log(f"saidas -> {OUT}")
print("TILE21KVT_2023_OK")
