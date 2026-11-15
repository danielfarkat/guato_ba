#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eficiência MÊS A MÊS (jun..nov) por tile, ancorando no MELHOR dia (maior cobertura limpa) do mês anterior.
Mês M: t1 = melhor(M-1); t2 = cada dia de M com cobertura>=COV; união das detecções = queimada de M.
Compara vs HLS mensal. Salva em dados/<tile>/: rasters mensais (modelo e HLS), PNG de concordância por mês,
CSV de eficiência e figura resumo."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, torch, rasterio
from rasterio import Affine; from rasterio.crs import CRS
from concurrent.futures import ProcessPoolExecutor
from rasterio.features import rasterize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap; from matplotlib.patches import Patch
import geo_data_science as g
from geo_data_science import grid_5880, geom_por_tile, _slot_do_tile, buscar_itens, carregar_hls_mes, PX_HA
import accumulate_tile as AT
from redes import construir_dl
CLOUD=20.0; COV=0.75; THR=0.90; INI="2024-01-01"; FIM="2024-12-31"; NPROC=5
DEV=torch.device("cuda")

def _save(path, arr, st, sh, sw, dtype="uint8", nodata=0):
    prof=dict(driver="GTiff",height=sh,width=sw,count=1,dtype=dtype,crs=CRS.from_epsg(5880),transform=st,
              nodata=nodata,compress="DEFLATE",tiled=True,blockxsize=256,blockysize=256)
    with rasterio.open(path,"w",**prof) as d: d.write(arr.astype(dtype),1)

def _save_fc(path, R, mask, st, sh, sw):
    """Falsa-cor p/ cicatriz: banda1=SWIR2(B12), banda2=NIR(B8A), banda3=Red(B04). uint16 DN=refl*10000, fora do bioma=0."""
    dn = np.clip(np.stack([R[3], R[1], R[0]], 0) * 10000, 0, 65535).astype(np.uint16)  # [B12,B8A,B04]
    dn[:, ~mask] = 0
    prof = dict(driver="GTiff", height=sh, width=sw, count=3, dtype="uint16", crs=CRS.from_epsg(5880),
                transform=st, nodata=0, compress="DEFLATE", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as d:
        d.write(dn)
        d.set_band_description(1, "SWIR2_B12"); d.set_band_description(2, "NIR_B8A"); d.set_band_description(3, "RED_B04")

def _concord_png(path, model, hls, dom, title):
    cat=np.full(model.shape,np.nan); cat[dom]=0; cat[(model&hls)]=1; cat[(~model&hls&dom)]=2; cat[(model&~hls&dom)]=3
    fig,ax=plt.subplots(figsize=(7,7))
    ax.imshow(cat,cmap=ListedColormap(["#ececec","#2ca02c","#ffcc00","#d62728"]),vmin=0,vmax=3,interpolation="nearest")
    ax.set_title(title,fontsize=10,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(color="#2ca02c",label="ambos"),Patch(color="#ffcc00",label="só HLS (omissão)"),
                       Patch(color="#d62728",label="só modelo (comissão)")],loc="lower left",fontsize=8,frameon=True)
    fig.tight_layout(); fig.savefig(path,dpi=130,bbox_inches="tight"); plt.close(fig)

def run_tile(modelo, tile):
    OUT=f"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba/dados/{tile.lower()}"; os.makedirs(OUT,exist_ok=True)
    ck=torch.load(f"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba/deeplearning/{modelo}/{modelo}_best.pkl",map_location=DEV,weights_only=False)
    bandas=ck["bandas"]; mean=ck["mean"]; std=ck["std"]
    model=construir_dl(modelo,p=0.0,pretrained=False).to(DEV); model.load_state_dict(ck["state_dict"]); model.eval()
    GH,GW,GGT=grid_5880(); geoms=geom_por_tile()
    r0,c0,sh,sw,st=_slot_do_tile(geoms[tile])
    tile_mask=rasterize([(geoms[tile],1)],out_shape=(sh,sw),transform=st,fill=0,dtype="uint8").astype(bool)
    area=int(tile_mask.sum())
    HLSm=carregar_hls_mes()[r0:r0+sh,c0:c0+sw]
    _save(f"{OUT}/hls_mes_{tile}.tif", np.where(tile_mask,HLSm,0), st, sh, sw)   # HLS mensal recortado no tile
    itens=buscar_itens(tile,INI,FIM,CLOUD)
    pordia={}
    for it in itens: pordia.setdefault(it.properties["datetime"][:10],[]).append(it)
    st_t=(st.a,st.b,st.c,st.d,st.e,st.f); dias=sorted(pordia)
    payloads=[([{b:it.assets[b].href for b in ("B04","B8A","B11","B12","SCL") if b in it.assets} for it in pordia[d]],st_t,sh,sw) for d in dias]
    comp={}; cov={}
    pool=ProcessPoolExecutor(max_workers=NPROC)
    for d,(R,val) in zip(dias,pool.map(AT._compose_worker,payloads)):
        vt=val&tile_mask; cc=vt.sum()/max(area,1)
        # salva CADA data como 21kwb_DD_MM_YYYY.tif (falsa-cor B12/B8A/B04)
        _save_fc(f"{OUT}/{tile.lower()}_{d[8:10]}_{d[5:7]}_{d[0:4]}.tif", R, tile_mask, st, sh, sw)
        if cc>=COV: comp[d]=(R,vt); cov[d]=cc
    pool.shutdown()
    g.log(f"[{tile}] imagens falsa-cor salvas: {len(dias)} datas (nuvem<{CLOUD}%)")
    best={}
    for d in comp:
        m=int(d[5:7])
        if m not in best or cov[d]>cov[best[m]]: best[m]=d
    g.log(f"[{tile}] datas limpas(>= {COV:.0%}, nuvem<{CLOUD}%): {len(comp)} | melhor/mês: "+", ".join(f"{m}:{best[m][5:]}({cov[best[m]]:.0%})" for m in sorted(best)))
    rows=[]; ACUM=np.zeros((sh,sw),bool)
    for M in range(6,12):
        if (M-1) not in best:
            g.log(f"  mês {M}: sem melhor dia em {M-1} — pulado"); continue
        datasM=[d for d in comp if int(d[5:7])==M]
        if not datasM:
            g.log(f"  mês {M}: sem datas limpas — pulado"); continue
        R1,v1=comp[best[M-1]]
        burned=np.zeros((sh,sw),bool); dom=np.zeros((sh,sw),bool)
        for d in datasM:
            R2,v2=comp[d]; dd=v1&v2&tile_mask
            if not dd.any(): continue
            prob=AT._infere_par(model,bandas,mean,std,R1,R2,dd,sh,sw)
            burned|=(prob>=THR)&dd; dom|=dd
        A=(HLSm==M)&tile_mask
        a=float((burned&A&dom).sum());b=float((burned&~A&dom).sum());c=float((~burned&A&dom).sum())
        f1=2*a/(2*a+b+c) if(2*a+b+c)else 0;iou=a/(a+b+c) if(a+b+c)else 0
        oe=c/(a+c) if(a+c)else 0;ce=b/(a+c) if(a+c)else 0; covpct=dom.sum()/max(area,1)*100
        # salva raster do mês + concordância
        _save(f"{OUT}/queimada_mes{M:02d}_{modelo}.tif", (burned&dom), st, sh, sw)
        _concord_png(f"{OUT}/concordancia_mes{M:02d}.png", burned&dom, A, dom,
                     f"{tile} mês {M:02d} (t1={best[M-1][5:]}) | IoU={iou:.2f} OE={oe:.2f} CE={ce:.2f}")
        ACUM|=(burned&dom)
        rows.append(dict(mes=M,t1=best[M-1],n_t2=len(datasM),cobertura_pct=round(covpct,1),
                         modelo_ha=round((burned&dom).sum()*PX_HA),hls_ha=round((A&dom).sum()*PX_HA),
                         F1=round(f1,3),IoU=round(iou,3),OE=round(oe,3),CE=round(ce,3)))
        g.log(f"  mês {M:>2} (t1={best[M-1][5:]}, {len(datasM)} t2, cob {covpct:.0f}%): modelo {(burned&dom).sum()*PX_HA:,.0f}ha HLS {(A&dom).sum()*PX_HA:,.0f}ha | F1={f1:.3f} IoU={iou:.3f} OE={oe:.3f} CE={ce:.3f}")
    _save(f"{OUT}/acumulado_jun_nov_{modelo}.tif", ACUM, st, sh, sw)
    import pandas as pd
    df=pd.DataFrame(rows); df.to_csv(f"{OUT}/eficiencia_mensal_{tile}.csv",index=False)
    # figura resumo
    if len(df):
        fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(df))
        ax.bar(x-0.2,df.hls_ha/1e3,0.4,label="HLS",color="#999"); ax.bar(x+0.2,df.modelo_ha/1e3,0.4,label="modelo",color="#1f77b4")
        ax2=ax.twinx(); ax2.plot(x,df.IoU,"o-",color="#d62728",label="IoU"); ax2.set_ylim(0,1); ax2.set_ylabel("IoU")
        ax.set_xticks(x); ax.set_xticklabels([f"mês {m}" for m in df.mes]); ax.set_ylabel("mil ha"); ax.legend(loc="upper left")
        ax.set_title(f"{tile} — eficiência mês a mês (modelo {modelo} vs HLS, cloud<{CLOUD:.0f}% cob>={COV:.0%})",fontweight="bold")
        fig.tight_layout(); fig.savefig(f"{OUT}/resumo_mensal_{tile}.png",dpi=130); plt.close(fig)
    g.log(f"✅ salvo em {OUT} | CSV + {len(df)} meses (rasters .tif + concordancia .png) + resumo")
    return df

if __name__=="__main__":
    run_tile(sys.argv[1] if len(sys.argv)>1 else "unet", sys.argv[2] if len(sys.argv)>2 else "21KWB")
