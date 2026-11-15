#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figura: 3 maiores polígonos de OMISSÃO + 3 de COMISSÃO (U-Net 128x753 vs HLS, 2024, área útil).
Painel por polígono: [T1 | T2 | Diferença(matriz par1) | HLS | Modelo].
- T1/T2 = cubos do par 1 (30/05, 19/07), falsa-cor SWIR2/NIR/SWIR1 com stretch min-max FIXO.
- Diferença = matriz_2024_par1 (R=dNDVI banda3, G=NBR_t2 banda2, B=dNBR banda1), stretch min-max FIXO.
- Máscaras HLS/modelo: preto=não queimada, branco=queimada.
Reusa os 6 polígonos já salvos em poligonos_3x3.gpkg."""
import os
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, rasterio, geopandas as gpd
from rasterio.windows import from_bounds
from affine import Affine
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
import geo_data_science as g
from geo_data_science import grid_5880, mascara_bioma, carregar_hls_mes, PX_HA, RES, ESC

# polígonos que usam datas FRESCAS do BDC (<10% nuvem de cena + cena mais limpa SOBRE o polígono)
OVERRIDE={("comissao",2):dict(tile="21KUU",t1_date="2024-07-19",t2_date="2024-11-16"),
          ("omissao",2): dict(tile="21KVB",win1=("2024-04-01","2024-05-10"),t2_date="2024-11-11"),
          ("omissao",3): dict(tile="21KVB",win1=("2024-04-01","2024-05-10"),t2_date="2024-11-11")}

OUT=f"{g.GUATO}/results/artigo/figura_poligonos"
GH,GW,GGT=grid_5880(); GX0,GYt=GGT.c,GGT.f; BIOMA=mascara_bioma()
with rasterio.open(f"{g.GUATO}/results/acumulado/acum2024_unet_t3_753.tif") as s: model=s.read(1)>0
with rasterio.open(f"{g.GUATO}/results/acumulado/analisado_unet_t3_753.tif") as s: anal=s.read(1)>0
DOM=anal&BIOMA; model=model&DOM; hls=(carregar_hls_mes()>0)&DOM
gdf=gpd.read_file(f"{OUT}/poligonos_3x3.gpkg").sort_values(["tipo","rank"]).reset_index(drop=True)
g.log(f"FIGURA par1 | {len(gdf)} polígonos")

from rasterio.features import rasterize
MR=f"{g.GUATO}/results/matriz_rgb"
CUB={d:f"{g.DADOS}/cubo_2024{d}_corrigido.tif" for d in ("0530","0719","0803","0902")}
# par -> (matriz, data_t1, data_t2)
PARS={"par1":(f"{MR}/matriz_2024_par1_t1t2_0530_0719.tif","0530","0719"),
      "par2":(f"{MR}/matriz_2024_par2_t2t4_0719_0902.tif","0719","0902"),
      "par3":(f"{MR}/matriz_2024_par3_t3t4_0803_0902.tif","0803","0902")}
def strm(a,lo,hi): return np.clip((a.astype(np.float32)-lo)/(hi-lo),0.0,1.0)
def rd(path,bounds,bands):
    with rasterio.open(path) as s:
        w=from_bounds(*bounds,transform=s.transform)
        return s.read(bands,window=w,boundless=True,fill_value=0)
def par_rgb(p1,p2,bounds):      # SWIR2(R)/NIR b8a(G)/Red b04(B); stretch percentil 3-96 das DUAS datas juntas
    a=rd(p1,bounds,[4,2,1]).astype(np.float32); b=rd(p2,bounds,[4,2,1]).astype(np.float32)
    o1=[]; o2=[]
    for ch in range(3):
        v=np.concatenate([a[ch][a[ch]>0], b[ch][b[ch]>0]])
        lo,hi=(np.percentile(v,3),np.percentile(v,96)) if v.size else (0,1)
        o1.append(strm(a[ch],lo,hi)); o2.append(strm(b[ch],lo,hi))
    return np.dstack(o1), np.dstack(o2)
def matriz_rgb(path,bounds):    # band3 dNDVI(R), band2 NBR_t2(G), band1 dNBR(B); int16 x10000
    a=rd(path,bounds,[3,2,1]).astype(np.float32)/10000.0
    return np.dstack([strm(a[0],-0.00770226,0.224989),strm(a[1],-0.1856,0.345397),strm(a[2],0.00359794,0.325298)])
def mean_dnbr(path,poly):       # dNBR médio (banda1) DENTRO do polígono
    with rasterio.open(path) as s:
        w=from_bounds(*poly.bounds,transform=s.transform).round_offsets().round_lengths()
        a=s.read(1,window=w,boundless=True,fill_value=-32768).astype(np.float32)/10000.0
        tr=s.window_transform(w)
    m=rasterize([(poly,1)],out_shape=a.shape,transform=tr,fill=0,dtype='uint8').astype(bool)
    v=a[m&(a>-3.0)]; return float(np.nanmean(v)) if v.size else -9.0
def melhor_par(poly):
    sc={p:mean_dnbr(PARS[p][0],poly) for p in PARS}
    return max(sc,key=sc.get), sc
def _slot(bounds):
    minx,miny,maxx,maxy=bounds
    sw=int(round((maxx-minx)/RES)); sh=int(round((maxy-miny)/RES))
    return Affine(RES,0,minx,0,-RES,maxy),sh,sw
def _clear_frac(item,poly,st,sh,sw):  # fração SCL-limpa DENTRO do polígono
    scl=g._reproj_banda(item.assets["SCL"].href,st,sh,sw)
    m=rasterize([(poly,1)],out_shape=(sh,sw),transform=st,fill=0,dtype='uint8').astype(bool)
    return float(np.isin(scl,list(g.SCL_KEEP))[m].mean()) if m.any() else 0.0
def pick_item(tile,win,poly,bounds,topn=10,minclear=0.80):  # <10% cena; limpa>=80% sobre o polígono -> a + recente
    cands=sorted(g.buscar_itens(tile,win[0],win[1],10.0),key=lambda x:x.properties.get("eo:cloud_cover",100))[:topn]
    st,sh,sw=_slot(bounds); scored=[]
    for it in cands:
        d=it.properties["datetime"][:10]; scored.append((it,d,_clear_frac(it,poly,st,sh,sw)))
    if not scored: raise RuntimeError(f"sem cena <10% {tile} {win}")
    good=[x for x in scored if x[2]>=minclear]
    return max(good,key=lambda x:x[1]) if good else max(scored,key=lambda x:x[2])
def read_item(item,bounds):          # B04/B8A/B12 (DN) reprojetado p/ janela 5880
    st,sh,sw=_slot(bounds)
    return (g._reproj_banda(item.assets["B04"].href,st,sh,sw).astype(np.float32),
            g._reproj_banda(item.assets["B8A"].href,st,sh,sw).astype(np.float32),
            g._reproj_banda(item.assets["B12"].href,st,sh,sw).astype(np.float32))
def fc_pair(A,B):                    # SWIR2(R)/NIR(G)/Red(B), stretch percentil 3-96 das duas datas
    o1=[];o2=[]
    for a,b in [(A[2],B[2]),(A[1],B[1]),(A[0],B[0])]:
        v=np.concatenate([a[a>0],b[b>0]]); lo,hi=(np.percentile(v,3),np.percentile(v,96)) if v.size else (0,1)
        o1.append(strm(a,lo,hi)); o2.append(strm(b,lo,hi))
    return np.dstack(o1),np.dstack(o2)
def diff_rgb(A,B):                   # dNDVI(R)/NBR_t2(G)/dNBR(B) calculados na hora, mesmos stretches do matriz
    r1,n1,s1=[x/ESC for x in A]; r2,n2,s2=[x/ESC for x in B]
    ndvi1=(n1-r1)/(n1+r1+1e-6); ndvi2=(n2-r2)/(n2+r2+1e-6); dndvi=ndvi1-ndvi2
    nbr1=(n1-s1)/(n1+s1+1e-6); nbr2=(n2-s2)/(n2+s2+1e-6); dnbr=nbr1-nbr2
    return np.dstack([strm(dndvi,-0.00770226,0.224989),strm(nbr2,-0.1856,0.345397),strm(dnbr,0.00359794,0.325298)])

n=len(gdf); fig,axes=plt.subplots(n,5,figsize=(16,3.1*n))
dnum={"omissao":"OMISSÃO","comissao":"COMISSÃO"}
for i,row in gdf.iterrows():
    poly=row.geometry; minx,miny,maxx,maxy=poly.bounds; pad=1500
    bnds=(minx-pad,miny-pad,maxx+pad,maxy+pad)
    c0=max(0,int((bnds[0]-GX0)/RES)); c1=min(GW,int(np.ceil((bnds[2]-GX0)/RES)))
    r0=max(0,int((GYt-bnds[3])/RES)); r1=min(GH,int(np.ceil((GYt-bnds[1])/RES)))
    mext=[GX0+c0*RES,GX0+c1*RES,GYt-r1*RES,GYt-r0*RES]
    dl={"0530":"30/05","0719":"19/07","0803":"03/08","0902":"02/09"}
    def rd_cube(d): a=rd(CUB[d],bnds,[1,2,4]).astype(np.float32); return (a[0],a[1],a[2])  # (red,nir,swir2)
    key=(row.tipo,int(row['rank']))
    if key in OVERRIDE:
        ov=OVERRIDE[key]
        def _forcar(date):   # cena exata de uma data (menor nuvem do dia), ignora filtro
            its=[x for x in g.buscar_itens(ov["tile"],date,date,100.0) if x.properties["datetime"][:10]==date]
            it=sorted(its,key=lambda x:x.properties.get("eo:cloud_cover",100))[0]
            st,sh,sw=_slot(bnds); return it,date,_clear_frac(it,poly,st,sh,sw)
        it1,da1,f1=_forcar(ov["t1_date"]) if ov.get("t1_date") else pick_item(ov["tile"],ov["win1"],poly,bnds)
        it2,da2,f2=_forcar(ov["t2_date"]) if ov.get("t2_date") else pick_item(ov["tile"],ov["win2"],poly,bnds)
        A=read_item(it1,bnds); Bq=read_item(it2,bnds)
        dtt1=f"{da1[8:10]}/{da1[5:7]}"; dtt2=f"{da2[8:10]}/{da2[5:7]}"
        g.log(f"  {row.tipo}#{row['rank']} BDC {ov['tile']} T1={da1}(limpo {f1:.0%}) T2={da2}(limpo {f2:.0%})")
    else:
        par,sc=melhor_par(poly); d1,d2=PARS[par][1],PARS[par][2]
        A=rd_cube(d1); Bq=rd_cube(d2); dtt1,dtt2=dl[d1],dl[d2]
        g.log("  %s#%d (%.0fha,%s): %s (%s->%s) | dNBR %s" % (row.tipo,row['rank'],row.area_ha,row.tile,par,dl[d1],dl[d2],{k:round(v,3) for k,v in sc.items()}))
    t1,t2=fc_pair(A,Bq); dif=diff_rgb(A,Bq)   # diferença SEMPRE calculada na hora (dNDVI/NBR_t2/dNBR)
    hh=hls[r0:r1,c0:c1]; mm=model[r0:r1,c0:c1]
    COLT=["(a) T1 — SWIR2/NIR8A/Red","(b) T2 — SWIR2/NIR8A/Red",
          "(c) Diferença — dNDVI/NBR$_{t2}$/dNBR","(d) Acumulado HLS","(e) Acumulado modelo"]
    panels=[(t1,bnds,None,dtt1),(t2,bnds,None,dtt2),(dif,bnds,None,None),
            (hh,mext,"gray",None),(mm,mext,"gray",None)]
    g.log("    brilho painéis: "+", ".join(f"{'abcde'[j]}={float(np.asarray(p[0]).mean()):.3f}" for j,p in enumerate(panels)))
    xs,ys=poly.exterior.xy; coords=np.column_stack([xs,ys])
    for j,(img,ext,cmap,dtxt) in enumerate(panels):
        ax=axes[i,j] if n>1 else axes[j]
        if cmap is None: ax.imshow(img,extent=[ext[0],ext[2],ext[1],ext[3]],origin="upper")
        else: ax.imshow(img,cmap="gray",vmin=0,vmax=1,extent=ext,origin="upper")
        ax.add_patch(MplPoly(coords,closed=True,facecolor=(1,1,0,0.18),edgecolor="yellow",lw=1.3,hatch="//"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor("black"); sp.set_linewidth(1.4)
        if dtxt: ax.text(0.04,0.96,dtxt,transform=ax.transAxes,fontsize=8,color="yellow",va="top",ha="left",
                         bbox=dict(boxstyle="round,pad=0.15",fc="black",ec="none",alpha=0.5))
        if i==0: ax.set_title(COLT[j],fontsize=8.5)
        if j==0: ax.set_ylabel(f"{dnum[row.tipo]} #{row['rank']}\n{row.area_ha:,.0f} ha\n{row.tile}",fontsize=8)
plt.tight_layout(); figpng=f"{OUT}/figura_3x3_poligonos.png"; plt.savefig(figpng,dpi=140,bbox_inches="tight"); plt.close()
g.log(f"figura -> {figpng}"); print("FIGURA_OK")
