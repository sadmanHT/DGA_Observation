#!/usr/bin/env python3
"""Regenerate the current paper's six included figures as native vector PDFs.

All plotted values are read from repository evidence; no values are hard-coded except
for the canonical confusion matrix, which is reconstructed from the canonical
classification result counts archived by the paper.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY="#163A5F"; BLUE="#2E5F8A"; MID="#5A86AD"; LIGHT="#8FB3D1"; PALE="#DCEAF7"
GRID="#D9E2EC"; TEXT="#1F2933"; MUTED="#5B6770"; WHITE="#FFFFFF"
plt.rcParams.update({
 "font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
 "font.size":8.2,"axes.labelsize":8.5,"xtick.labelsize":7.6,"ytick.labelsize":7.6,
 "legend.fontsize":7.3,"axes.edgecolor":TEXT,"axes.linewidth":0.75,"pdf.fonttype":42,"ps.fonttype":42,
})
ONECOL=3.45; TWOCOL=7.10

def clean(ax, grid="y"):
 ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
 if grid: ax.grid(axis=grid,color=GRID,lw=.55); ax.set_axisbelow(True)
 ax.tick_params(length=3,width=.65)

def save(fig,name):
 fig.savefig(OUT/name,format="pdf",bbox_inches="tight",pad_inches=.025)
 plt.close(fig)

# Fig. 1 adaptive pipeline
fig,ax=plt.subplots(figsize=(TWOCOL,1.70)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
boxes=[(.015,.42,.16,.37,"Observe current\nDGA prefix"),(.215,.42,.16,.37,"Extract compact\ntemporal features"),
       (.415,.42,.16,.37,"Predict class\nand score"),(.615,.42,.16,.37,"Persistent\nhigh confidence?")]
for x,y,w,h,t in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.01,rounding_size=.008",ec=NAVY,fc=PALE,lw=1.2))
 ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=8.5)
ax.add_patch(FancyBboxPatch((.825,.58),.16,.25,boxstyle="round,pad=.01,rounding_size=.008",ec=NAVY,fc=PALE,lw=1.2))
ax.text(.905,.705,"STOP\nissue diagnosis",ha="center",va="center",fontsize=8.5,fontweight="bold")
ax.add_patch(FancyBboxPatch((.825,.16),.16,.25,boxstyle="round,pad=.01,rounding_size=.008",ec=BLUE,fc=WHITE,lw=1.2))
ax.text(.905,.285,"CONTINUE\nor defer at endpoint",ha="center",va="center",fontsize=8.2)
for i in range(3):
 x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
 ax.add_patch(FancyArrowPatch((x1+.005,.605),(x2-.005,.605),arrowstyle="->",mutation_scale=11,lw=1.25,color=NAVY))
ax.add_patch(FancyArrowPatch((.775,.63),(.825,.705),arrowstyle="->",mutation_scale=11,lw=1.25,color=NAVY)); ax.text(.797,.704,"yes",fontsize=7.2,color=MUTED)
ax.add_patch(FancyArrowPatch((.695,.42),(.825,.285),arrowstyle="->",mutation_scale=11,lw=1.25,color=BLUE)); ax.text(.765,.36,"no",fontsize=7.2,color=MUTED)
ax.add_patch(FancyArrowPatch((.825,.23),(.095,.42),connectionstyle="arc3,rad=-.25",arrowstyle="->",mutation_scale=10,lw=1,color=LIGHT))
ax.text(.44,.08,"next observation horizon",fontsize=7.4,color=MUTED,ha="center")
save(fig,"adaptive_pipeline.pdf")

# Fig. 2 canonical confusion matrix: vector rectangles, no raster image/colorbar.
cm=np.array([[713,0,6,12],[0,34,0,4],[4,0,42,3],[4,2,1,75]],float); cls=["Normal","PD","Low-E","Overheat"]
fig,ax=plt.subplots(figsize=(ONECOL,3.05)); mx=cm.max()
import matplotlib.colors as mcolors
def mix(a,b,t):
 a=np.array(mcolors.to_rgb(a)); b=np.array(mcolors.to_rgb(b)); return mcolors.to_hex(a*(1-t)+b*t)
for i in range(4):
 for j in range(4):
  t=float(cm[i,j]/mx); fc=mix("#F8FBFE",NAVY,t**.55)
  ax.add_patch(Rectangle((j,i),1,1,facecolor=fc,edgecolor=WHITE,lw=.7))
  ax.text(j+.5,i+.5,str(int(cm[i,j])),ha="center",va="center",fontsize=8.6,fontweight="bold",color=WHITE if t>.42 else TEXT)
ax.set_xlim(0,4); ax.set_ylim(4,0); ax.set_aspect("equal"); ax.set_xticks(np.arange(4)+.5,cls,rotation=35,ha="right"); ax.set_yticks(np.arange(4)+.5,cls)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.tick_params(length=0)
save(fig,"confusion_matrix_final.pdf")

# Fig. 3 dense horizon representation
df=pd.read_csv(E/"advanced/dense_representation/dense_representation_curve.csv")
fig,ax=plt.subplots(figsize=(ONECOL,2.55))
for rep,c,m,ls,label in [("Statistical_28",NAVY,"o","-","Statistical-28"),("Temporal_82",MID,"s","--","Temporal-82")]:
 d=df[df.representation==rep].sort_values("horizon_percent"); ax.plot(d.horizon_percent,d.test_macro_f1,color=c,marker=m,ls=ls,lw=1.35,ms=3.3,markevery=2,label=label)
ax.axvspan(55,75,color=PALE,lw=0); ax.set_xlabel("Available DGA history (%)"); ax.set_ylabel("Test macro-F1"); ax.set_xlim(10,100); ax.set_xticks([10,25,50,75,100]); ax.set_ylim(.66,.95); clean(ax,"both"); ax.legend(frameon=False,loc="upper left")
save(fig,"dense_horizon_representation.pdf")

# Fig. 4 adaptive stopping distribution
df=pd.read_csv(E/"advanced/adaptive/adaptive_policy_test_per_case.csv"); counts=df.stop_horizon_percent.value_counts().sort_index(); hs=list(range(10,101,5)); vals=[int(counts.get(h,0)) for h in hs]
fig,ax=plt.subplots(figsize=(ONECOL,2.5)); ax.bar(hs,vals,width=3.9,color=BLUE,edgecolor=NAVY,lw=.35); ax.set_xlabel("Stopping history (%)"); ax.set_ylabel("Test cases"); ax.set_xlim(8,102); ax.set_xticks([10,20,40,60,80,100]); clean(ax,"y"); save(fig,"adaptive_stop_distribution.pdf")

# Fig. 5 SHAP family stability
df=pd.read_csv(E/"advanced/shap/shap_grouped.csv"); d=df[df.group_type=="family"]
fams=["statistical","endpoint_change","local_dynamics","phase_shift","trend","cross_gas_correlation"]
labels={"statistical":"Statistical","endpoint_change":"Endpoint/change","local_dynamics":"Local dynamics","phase_shift":"Phase shift","trend":"Trend","cross_gas_correlation":"Cross-gas corr."}
fig,ax=plt.subplots(figsize=(ONECOL,2.85)); y=np.arange(len(fams)); bh=.22
for off,h,c in zip([-bh,0,bh],[50,75,100],[LIGHT,MID,NAVY]):
 vals=[float(d[(d.horizon_percent==h)&(d.group==f)].share_of_total_importance.iloc[0]) for f in fams]; ax.barh(y+off,vals,height=bh*.92,color=c,label=f"{h}%")
ax.set_yticks(y,[labels[f] for f in fams]); ax.invert_yaxis(); ax.set_xlabel("Share of total mean |SHAP|"); clean(ax,"x"); ax.tick_params(axis="y",length=0); ax.legend(frameon=False,loc="lower right",ncol=3,columnspacing=.8); save(fig,"shap_family_by_horizon.pdf")

# Fig. 6 robustness refinement
noise=pd.read_csv(E/"refinement/04_noise_test_summary.csv"); clean_df=pd.read_csv(E/"refinement/04_clean_test_comparison.csv")
styles=[("Temporal-82 original",LIGHT,"o",":","Temporal-82"),("Temporal-70 unsmoothed",MID,"s","--","Temporal-70"),("Temporal-70 robust w=11",NAVY,"^","-","Robust Temporal-70")]
fig,ax=plt.subplots(figsize=(ONECOL,2.65))
for model,c,m,ls,label in styles:
 cv=float(clean_df[clean_df.model==model].macro_f1.iloc[0]); z=noise[noise.model==model].sort_values("severity"); xs=np.r_[0,z.severity.to_numpy()*100]; ys=np.r_[cv,z.macro_f1_mean.to_numpy()]
 ax.plot(xs,ys,color=c,marker=m,ls=ls,lw=1.35,ms=3.8,label=label); ax.errorbar(z.severity*100,z.macro_f1_mean,yerr=z.macro_f1_std,fmt="none",ecolor=c,elinewidth=.8,capsize=2)
ax.set_xlabel("Independent multiplicative noise (%)"); ax.set_ylabel("Test macro-F1"); ax.set_xticks([0,2,5,10]); ax.set_ylim(.18,.93); clean(ax,"both"); ax.legend(frameon=False,loc="upper right"); save(fig,"robustness_refinement.pdf")

print("Vector paper figures written to", OUT)
