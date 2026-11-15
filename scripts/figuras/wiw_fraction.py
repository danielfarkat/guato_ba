#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantifica a fração do BIOMA removida pelo WIW ALÉM da SCL.
Para cada data 2024: SCL-válido = bandas>0 ∩ SCL∈SCL_KEEP ; água WIW = refl(B8A)<0.1804 & refl(B11)<0.1131.
'Removido pelo WIW além da SCL' = pixels SCL-válidos marcados como água (que a SCL manteria).
Reporta por data e média; frações sobre o bioma e sobre o SCL-válido."""
import os
for k in ("PROJ_DATA","PROJ_LIB"): os.environ[k]="/opt/conda/envs/geospatial/share/proj"
os.environ["GDAL_DATA"]="/opt/conda/envs/geospatial/share/gdal"
import sys; sys.path.insert(0,"/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba")
import numpy as np, rasterio, pandas as pd
import geo_data_science as g
from geo_data_science import grid_5880, mascara_bioma, refl, WIW_NIR, WIW_SW1, SCL_KEEP, PX_HA, BAND, CUBOS_2024

B = mascara_bioma(); nb = int(B.sum()); A_bioma = nb*PX_HA
g.log(f"WIW além da SCL | bioma={A_bioma:,.0f} ha | SCL_KEEP={sorted(SCL_KEEP)} | WIW NIR<{WIW_NIR} SWIR1<{WIW_SW1}")
rows=[]
for nome,path in CUBOS_2024.items():
    with rasterio.open(path) as s:
        b8a=s.read(BAND["b8a"]); b11=s.read(BAND["b11"]); b12=s.read(BAND["b12"]); scl=s.read(BAND["scl"])
    scl_ok = (b8a>0)&(b11>0)&(b12>0)&np.isin(scl,SCL_KEEP)&B    # válido p/ SCL (antes do WIW)
    agua   = (refl(b8a)<WIW_NIR)&(refl(b11)<WIW_SW1)
    rem    = scl_ok & agua                                       # SCL manteria, WIW remove (água)
    n_scl=int(scl_ok.sum()); n_rem=int(rem.sum())
    fr_bioma = n_rem/nb*100; fr_scl = (n_rem/n_scl*100) if n_scl else 0.0
    g.log(f"  {nome}: SCL-válido={n_scl*PX_HA:,.0f}ha | WIW remove(água)={n_rem*PX_HA:,.0f}ha | {fr_bioma:.2f}% do bioma | {fr_scl:.2f}% do SCL-válido")
    rows.append(dict(data=nome, scl_valido_ha=round(n_scl*PX_HA,1), wiw_remove_ha=round(n_rem*PX_HA,1),
                     frac_bioma_pct=round(fr_bioma,2), frac_scl_pct=round(fr_scl,2)))
    del b8a,b11,b12,scl,scl_ok,agua,rem
df=pd.DataFrame(rows)
mean=dict(data="MÉDIA", scl_valido_ha=round(df.scl_valido_ha.mean(),1), wiw_remove_ha=round(df.wiw_remove_ha.mean(),1),
          frac_bioma_pct=round(df.frac_bioma_pct.mean(),2), frac_scl_pct=round(df.frac_scl_pct.mean(),2))
df=pd.concat([df,pd.DataFrame([mean])],ignore_index=True)
out=f"{g.GUATO}/results/artigo/wiw_fracao_bioma.csv"; df.to_csv(out,index=False)
g.log(f"CSV -> {out}")
print(df.to_string(index=False)); print("WIW_FRACAO_OK")
