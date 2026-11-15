#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PNG comparativo: validação vs teste OOT 2025 em 3 regimes de validação.
Mostra o quão bem (ou mal) a VALIDAÇÃO prevê o desempenho real no ano de fora (OOT 2025)."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
OUT="/home/jovyan/ID-QUEIMADAS/jupyter/guato_ba/results/artigo"
# (rótulo, IoU validação, IoU OOT2025, nota)
cen=[("Com vazamento\n(split aleatório por patch)", 0.96, 0.64),
     ("Sem vazamento\n(split espacial N/S)",        0.894,0.838),
     ("Validação conhecida\n(2025 no treino, in-sample)", None, 0.768)]
labels=[c[0] for c in cen]; x=np.arange(len(cen)); w=0.36
val=[c[1] for c in cen]; oot=[c[2] for c in cen]
fig,ax=plt.subplots(figsize=(9,5.2))
b1=ax.bar(x-w/2,[v if v is not None else 0 for v in val],w,label="IoU validação",color="#4393C3")
b2=ax.bar(x+w/2,oot,w,label="IoU teste OOT 2025",color="#D6604D")
for i,(v,o) in enumerate(zip(val,oot)):
    if v is not None:
        ax.text(i-w/2,v+0.01,f"{v:.2f}",ha="center",fontsize=9)
        ax.annotate(f"gap {v-o:+.2f}",(i,max(v,o)+0.05),ha="center",fontsize=8,color="0.3")
    else:
        ax.text(i-w/2,0.02,"(val = teste:\nmodelo já viu 2025)",ha="center",va="bottom",fontsize=7.5,color="0.3")
    ax.text(i+w/2,o+0.01,f"{o:.2f}",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
ax.set_ylabel("IoU da classe queimada"); ax.set_ylim(0,1.08)
ax.set_title("Validação vs teste OOT 2025 — efeito do regime de validação\n(a validação só é confiável quando prevê o OOT)",fontsize=11)
ax.legend(loc="upper right",fontsize=9); ax.grid(axis="y",alpha=0.3)
ax.axhline(0,color="k",lw=0.6)
plt.tight_layout(); png=f"{OUT}/cenarios_validacao_oot.png"; plt.savefig(png,dpi=150); plt.close()
print("PNG ->",png)
