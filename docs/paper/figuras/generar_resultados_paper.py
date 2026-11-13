"""
Cifras, pruebas y figuras de la sección de Resultados del artículo.

Lee solo archivos de results/ (no entrena nada; tarda menos de un minuto) y escribe:
- results/paper_significancia.csv: Nadeau-Bengio, McNemar exacta y t bayesiana
  correlacionada (ROPE ±0.005) de DES-CPL frente a cada benchmark;
- results/paper_prueba_umbral.csv: métricas de prueba con el umbral del 15 %;
- results/paper_ablacion.csv y results/paper_rangos_conjuntos.csv;
- las figuras fig_*.pdf de docs/paper/figuras/ (vectoriales, en inglés).

Uso (raíz del repositorio): python docs/paper/figuras/generar_resultados_paper.py
"""
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(RAIZ))
from src.estadistica import prueba_mcnemar_exacta, t_bayesiana_correlacionada

# ---------- 1. Pruebas y métricas ----------
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
R=str(RAIZ/"results")+"/"
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
BASE=["logistica","bayes","knn","arbol","random_forest","xgboost","svm"]; SOTA=["lightgbm","catboost"]
BENCH=BASE+SOTA; EST=["Promedio simple","Apilamiento logístico","Pesos globales (Brier)"]
ev=pd.read_csv(R+"des_cpl_evaluacion_60.csv"); sota=pd.read_csv(R+"benchmarks_sota.csv").rename(columns={"modelo":"método"})
cv=pd.concat([ev[["fold","método","auc_pr","auc_roc","brier"]],sota[["fold","método","auc_pr","auc_roc","brier"]]])
pf=cv.pivot_table(index="fold",columns="método",values="auc_pr")
print("== CV")
tab=cv.groupby("método")[["auc_pr","auc_roc","brier"]].agg(["mean","std"])
print(tab.loc[BENCH+EST+["DES-CPL"]].round(4))
# pruebas por fold
def nb(d): 
    d=np.asarray(d); k=len(d); s=np.sqrt((1/k+0.25)*d.var(ddof=1)); t=d.mean()/s
    return t, stats.t.sf(t,k-1), s
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o): run=max(run,(m-i)*p[j]); adj[j]=min(1,run)
    return adj
def cliff(a,b):
    a=np.asarray(a);b=np.asarray(b); return (np.sign(a[:,None]-b[None,:])).mean()
ROPE=0.005
filas=[]
for b in BENCH:
    d=pf["DES-CPL"]-pf[b]; t,p,s=nb(d); tc=stats.t.ppf(0.975,4)
    bay=t_bayesiana_correlacionada(d,rope=ROPE)   # misma función que el capítulo 16
    filas.append(dict(benchmark=b,dif=d.mean(),ic_inf=d.mean()-tc*s,ic_sup=d.mean()+tc*s,t=t,p=p,
        P_mejor=bay["mejor"],P_rope=bay["equivalente"],P_peor=bay["peor"],cliff=cliff(pf["DES-CPL"],pf[b]),gana=int((d>0).sum())))
sig=pd.DataFrame(filas); sig["p_holm"]=holm(sig.p)
# test
pr=pd.read_csv(R+"prueba_predicciones.csv"); y=pr["y"].values
def umbral(p): return p>=np.quantile(p,0.85)
filas2=[]
for m in BENCH+EST+["DES-CPL"]:
    p=pr[m].values; yh=umbral(p)
    filas2.append(dict(método=m,auc_pr=average_precision_score(y,p),auc_roc=roc_auc_score(y,p),f1=f1_score(y,yh),mcc=matthews_corrcoef(y,yh),bacc=balanced_accuracy_score(y,yh),prec=precision_score(y,yh),rec=recall_score(y,yh)))
te=pd.DataFrame(filas2).set_index("método"); print("== prueba umbral 15%"); print(te.round(4))
# McNemar exact on correctness
mc=[]
yd=umbral(pr["DES-CPL"].values)
for b in BENCH:
    r=prueba_mcnemar_exacta(y,yd,umbral(pr[b].values))   # misma función que el capítulo 16
    mc.append(dict(benchmark=b,des_bien_b_mal=r["n01"],des_mal_b_bien=r["n10"],p_mcnemar=r["p"]))
mc=pd.DataFrame(mc); mc["p_mcnemar_holm"]=holm(mc.p_mcnemar)
dl=pd.read_csv(R+"../results/prueba_delong.csv") if False else None
sig=sig.merge(mc,on="benchmark"); print("== significancia"); print(sig.round(4))
sig.to_csv(R+"paper_significancia.csv",index=False); te.to_csv(R+"paper_prueba_umbral.csv")
# confusion DES vs XGB
for m in ["DES-CPL","xgboost"]:
    print(m, confusion_matrix(y,umbral(pr[m].values)).ravel())
# ablation
AB=[c for c in pf.columns if c.startswith("Ablación")]+["DES-CPL inicial (sin poda ni encogimiento)"]
fa=[]
for a in AB:
    d=pf["DES-CPL"]-pf[a]; t,p,s=nb(d); fa.append(dict(variante=a,media=pf[a].mean(),sd=pf[a].std(),dif=d.mean(),t=t,p=p,gana=int((d>0).sum())))
fa=pd.DataFrame(fa); fa["p_holm"]=holm(fa.p); print("== ablación"); print(fa.round(4)); fa.to_csv(R+"paper_ablacion.csv",index=False)
# lambda chosen
print(ev[ev["método"]=="DES-CPL"][["fold","k","tau","lambda","modelos_combinados","auc_pr"]])

# ---------- 2. Rangos entre conjuntos ----------
import numpy as np, pandas as pd
from scipy import stats
R=str(RAIZ/"results")+"/"
BENCH=["logistica","bayes","knn","arbol","random_forest","xgboost","svm","lightgbm","catboost"]
EST=["Promedio simple","Apilamiento logístico","Pesos globales (Brier)"]; MET=BENCH+EST+["DES-CPL"]
ev=pd.read_csv(R+"des_cpl_evaluacion_60.csv"); so=pd.read_csv(R+"benchmarks_sota.csv").rename(columns={"modelo":"método"})
p=pd.concat([ev,so]).groupby("método")["auc_pr"].mean()
ad=pd.read_csv(R+"adicionales/evaluacion_conjuntos.csv").groupby(["conjunto","método"])["auc_pr"].mean().unstack(0)
M=pd.concat([p.rename("diabetes"),ad],axis=1).loc[MET]
rk=M.rank(ascending=False); mr=rk.mean(1).sort_values(); print(mr)
k,N=len(MET),M.shape[1]
chi=12*N/(k*(k+1))*((mr**2).sum()-k*(k+1)**2/4); F=(N-1)*chi/(N*(k-1)-chi)
print("chi2",chi,"p",stats.chi2.sf(chi,k-1),"FF",F,"p",stats.f.sf(F,k-1,(k-1)*(N-1)))
q=stats.studentized_range.ppf(0.95,k,np.inf)/np.sqrt(2); print("CD",q*np.sqrt(k*(k+1)/(6*N)))
mr.to_csv(R+"paper_rangos_conjuntos.csv")
pt=pd.read_csv(R+"presupuesto_tiempos.csv"); print(pt)
print(pd.read_csv(R+"equidad_des_cpl_prueba.csv").head(40))
sh=pd.read_csv(R+"shap_importancia_global.csv",index_col=0); print(sh.sort_values("des_cpl",ascending=False).head(10))
print(pd.read_csv(R+"prueba_conformal.csv")); print(pd.read_csv(R+"prueba_resumen.csv"))

# ---------- 3. Figuras (formato de la guía: PDF vectorial, 6.3 in, 12-13 pt, sin título) ----------
RESULTADOS=str(RAIZ/'results')+'/'; FIGURAS=str(RAIZ/'docs'/'paper'/'figuras')+'/'
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
for f in ["lmsans10-regular.otf","lmsans10-bold.otf"]:
    try: fm.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/lm/"+f)
    except Exception: pass
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Latin Modern Sans","DejaVu Sans"],
 "font.size":12,"axes.labelsize":13,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,
 "lines.linewidth":2.0,"axes.grid":True,"grid.alpha":0.3,"pdf.fonttype":42,"savefig.bbox":"tight",
 "mathtext.fontset":"stixsans","legend.frameon":False,"axes.spines.top":False,"axes.spines.right":False})
R=RESULTADOS; O=FIGURAS
DES="#C1443C"; XGB="#2A86B0"; GRIS="#8a8a8a"; NAVY="#1B2A4E"; W=6.3
EN={"logistica":"Logistic reg.","bayes":"Naive Bayes","knn":"k-NN","arbol":"Decision tree","random_forest":"Random Forest","xgboost":"XGBoost","svm":"Linear SVM","lightgbm":"LightGBM","catboost":"CatBoost","Promedio simple":"Simple average","Apilamiento logístico":"Logistic stacking","Pesos globales (Brier)":"Global weights","DES-CPL":"DES-CPL"}
BENCH=["logistica","bayes","knn","arbol","random_forest","xgboost","svm","lightgbm","catboost"]
EST=["Promedio simple","Apilamiento logístico","Pesos globales (Brier)"]
def estilo(m):  # color, marcador: legible también en escala de grises
    if m=="DES-CPL": return DES,"s"
    if m=="xgboost": return XGB,"D"
    if m in EST: return NAVY,"^"
    return GRIS,"o"
ev=pd.read_csv(R+"des_cpl_evaluacion_60.csv"); so=pd.read_csv(R+"benchmarks_sota.csv").rename(columns={"modelo":"método"})
cv=pd.concat([ev[["fold","método","auc_pr"]],so[["fold","método","auc_pr"]]]); pf=cv.pivot_table(index="fold",columns="método",values="auc_pr")
# 1. AUC-PR por pliegue
orden=pf[BENCH+EST+["DES-CPL"]].mean().sort_values().index
fig,ax=plt.subplots(figsize=(W,5.4))
for i,m in enumerate(orden):
    c,mk=estilo(m); v=pf[m].values
    ax.scatter(v,np.full(5,i)+np.linspace(-.18,.18,5),s=22,color=c,marker=mk,alpha=.75,linewidths=0)
    ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt="|",color=c,ms=12,mew=2.5,capsize=3,lw=2)
ax.set_yticks(range(len(orden)),[EN[m] for m in orden]); ax.set_xlabel("AUC-PR on outer folds (dimensionless)")
ax.axvline(0.1139,color="#999999",lw=2,ls=":"); ax.text(0.1155,0.2,"prevalence",fontsize=12,color="#666666")
ax.grid(axis="y",visible=False); fig.savefig(O+"fig_folds_aucpr.pdf"); plt.close()
# 2. diferencias con IC
sig=pd.read_csv(R+"paper_significancia.csv").set_index("benchmark").loc[BENCH].sort_values("dif")
fig,ax=plt.subplots(figsize=(W,4.4))
ax.axvspan(-0.005,0.005,color="#e9dcc2",alpha=.8,lw=0,label="ROPE ($\\pm$0.005)")
ax.axvline(0,color="#555555",lw=1.5)
for i,(b,r) in enumerate(sig.iterrows()):
    ax.errorbar(r.dif,i,xerr=[[r.dif-r.ic_inf],[r.ic_sup-r.dif]],fmt="s",color=DES,ms=7,capsize=4,lw=2)
    ax.text(0.052,i,f"{r.p_holm:.3f}",va="center",fontsize=12,color="#333333")
ax.text(0.052,len(sig)-0.2,"p (Holm)",fontsize=12,color="#333333",va="bottom")
ax.set_yticks(range(len(sig)),[EN[b] for b in sig.index]); ax.set_xlim(-0.016,0.064)
ax.set_xlabel("$\\Delta$ AUC-PR, DES-CPL minus benchmark (95% CI)"); ax.grid(axis="y",visible=False)
ax.legend(loc="lower right",bbox_to_anchor=(0.80,0.0)); fig.savefig(O+"fig_diferencias_ic.pdf"); plt.close()
# 3. curvas de prueba + matriz de confusión
pr=pd.read_csv(R+"prueba_predicciones.csv"); y=pr["y"].values
fig,axs=plt.subplots(2,2,figsize=(W,6.2)); a=axs.ravel()
for m,c,ls,mk in [("xgboost",XGB,"-","o"),("DES-CPL",DES,"--","s")]:
    f,t,_=roc_curve(y,pr[m]); a[0].plot(f,t,color=c,ls=ls,label=EN[m])
    pp,rr,_=precision_recall_curve(y,pr[m]); a[1].plot(rr,pp,color=c,ls=ls,label=EN[m])
    o,pm=calibration_curve(y,pr[m],n_bins=10,strategy="quantile"); a[2].plot(pm,o,ls=ls,marker=mk,ms=5,color=c,label=EN[m])
a[0].plot([0,1],[0,1],color="#999999",ls=":",lw=1.5); a[0].set(xlabel="False positive rate",ylabel="True positive rate")
a[1].axhline(y.mean(),color="#999999",ls=":",lw=1.5); a[1].set(xlabel="Recall",ylabel="Precision",ylim=(0,0.6))
a[2].plot([0,.3],[0,.3],color="#999999",ls=":",lw=1.5); a[2].set(xlabel="Mean predicted prob.",ylabel="Observed rate",xlim=(0,.3),ylim=(0,.3))
for k in range(3): a[k].legend(loc="lower right" if k!=1 else "upper right",fontsize=11)
cm={m:confusion_matrix(y,pr[m].values>=np.quantile(pr[m].values,0.85)) for m in ["DES-CPL","xgboost"]}
a[3].imshow(cm["DES-CPL"]/cm["DES-CPL"].sum(1,keepdims=True),cmap="Greys",vmin=0,vmax=1.6)
for i in range(2):
    for j in range(2): a[3].text(j,i,f"{cm['DES-CPL'][i,j]:,}\n({cm['xgboost'][i,j]:,})",ha="center",va="center",fontsize=12)
a[3].set_xticks([0,1],["Not flagged","Flagged"]); a[3].set_yticks([0,1],["No readm.","Readm."]); a[3].grid(False)
a[3].set_xlabel("Prediction (top 15%)"); a[3].set_ylabel("Observed")
for k,l in enumerate("abcd"): a[k].text(-0.30 if k%2==0 else -0.28,1.06,f"({l})",transform=a[k].transAxes,fontsize=13,weight="bold")
fig.tight_layout(); fig.savefig(O+"fig_prueba_curvas.pdf"); plt.close()
# 4. ablación
ab=pd.read_csv(R+"paper_ablacion.csv")
NOM={"Ablación: Brier balanceado":"Class-balanced Brier","Ablación: competencia por exactitud":"Accuracy competence","Ablación: región sin estratificar":"Unstratified region","Ablación: selección dura (τ = 0)":"Hard selection ($\\tau$ = 0)","Ablación: sin encogimiento (λ = 0)":"No shrinkage ($\\lambda$ = 0)","Ablación: sin excluir al paciente":"Patient not excluded","Ablación: sin poda (7 modelos)":"No pruning (7 models)","DES-CPL inicial (sin poda ni encogimiento)":"Initial version"}
ab=ab.sort_values("dif"); sd=np.array([(pf["DES-CPL"]-pf[v]).std(ddof=1) for v in ab.variante])
fig,ax=plt.subplots(figsize=(W,4.4))
for i,(d,s) in enumerate(zip(ab.dif,sd)):
    ax.barh(i,d,xerr=s,height=.6,color=DES if d>0 else "#bbbbbb",hatch="" if d>0 else "//",edgecolor="#555555",lw=.8,error_kw=dict(lw=1.5,capsize=3,ecolor="#333333"))
ax.axvline(0,color="#555555",lw=1.5); ax.set_yticks(range(len(ab)),[NOM[v] for v in ab.variante])
ax.set_xlabel("$\\Delta$ AUC-PR, full model minus variant"); ax.grid(axis="y",visible=False)
fig.savefig(O+"fig_ablacion.pdf"); plt.close()
# 5. sensibilidad
se=pd.read_csv(R+"sensibilidad_des_cpl.csv"); g=se[se.barrido=="lambda x k"].pivot_table(index="lambda",columns="k",values="auc_pr")
fig,ax=plt.subplots(figsize=(W,4.2))
im=ax.imshow(g.values,cmap="Blues",aspect="auto",origin="lower")
ax.set_xticks(range(g.shape[1]),g.columns); ax.set_yticks(range(g.shape[0]),[f"{v:g}" for v in g.index]); ax.grid(False)
for i in range(g.shape[0]):
    for j in range(g.shape[1]): ax.text(j,i,f"{g.values[i,j]:.3f}",ha="center",va="center",fontsize=12,color="white" if g.values[i,j]>0.25 else "#1B2A4E")
ax.set(xlabel="k (neighbours per class)",ylabel="$\\lambda$ (shrinkage)")
cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03); cb.set_label("AUC-PR, fold 0")
fig.savefig(O+"fig_sensibilidad.pdf"); plt.close()
# 6. semillas
s=pd.read_csv(R+"semillas_des_cpl.csv"); ps=s.groupby(["semilla","método"])["auc_pr"].mean().unstack()
order=[1,2,3,4,5,6,7,8,42,2026]; x=np.arange(len(order))
fig,ax=plt.subplots(figsize=(W,3.8))
ax.plot(x,ps.loc[order,"xgboost"],"-o",color=XGB,ms=6,label="XGBoost")
ax.plot(x,ps.loc[order,"DES-CPL"],"--s",color=DES,ms=6,label="DES-CPL")
ax.set_xticks(x,[str(v) for v in order]); ax.set(xlabel="Random seed",ylabel="Mean AUC-PR (5 folds)")
ax.annotate("pre-registered seed",xy=(8,ps.loc[42,"xgboost"]),xytext=(4.6,0.2286),fontsize=12,arrowprops=dict(arrowstyle="-",color="#666666",lw=1))
ax.legend(loc="lower left"); fig.savefig(O+"fig_semillas.pdf"); plt.close()
# 7. rangos entre conjuntos
mr=pd.read_csv(R+"paper_rangos_conjuntos.csv",index_col=0).iloc[:,0].sort_values()
fig,ax=plt.subplots(figsize=(W,5.0))
for i,(m,v) in enumerate(mr.items()):
    c,_=estilo(m); h={"DES-CPL":"//","xgboost":".."}.get(m,"")
    ax.barh(i,v,color=c,height=.6,hatch=h,edgecolor="white" if h else c,lw=0)
    ax.text(v+.15,i,f"{v:.2f}",va="center",fontsize=12)
ax.set_yticks(range(len(mr)),[EN[m] for m in mr.index]); ax.invert_yaxis(); ax.grid(axis="y",visible=False)
cd=9.12; yb=len(mr)+0.1; ax.plot([1,1+cd],[yb,yb],color="#333333",lw=2); ax.text(1+cd+0.3,yb,f"CD = {cd:.2f}",va="center",fontsize=12)
ax.set_xlim(0,15); ax.set_ylim(len(mr)+0.7,-0.7); ax.set_xlabel("Mean rank of AUC-PR over 4 datasets (1 = best)")
fig.savefig(O+"fig_rangos_conjuntos.pdf"); plt.close()
# 8. SHAP
sh=pd.read_csv(R+"shap_importancia_global.csv",index_col=0); top=sh["des_cpl"].sort_values(ascending=False).head(10).index
fig,ax=plt.subplots(figsize=(W,4.8)); yy=np.arange(len(top))
ax.barh(yy-.2,sh.loc[top,"des_cpl"],height=.4,color=DES,hatch="//",edgecolor="white",lw=0,label="DES-CPL")
ax.barh(yy+.2,sh.loc[top,"xgboost"],height=.4,color=XGB,label="XGBoost")
ax.set_yticks(yy,top); ax.invert_yaxis(); ax.set_xlabel("Mean |SHAP| (probability points)"); ax.grid(axis="y",visible=False)
ax.legend(loc="lower right"); fig.savefig(O+"fig_shap_des_cpl.pdf"); plt.close()
print("figuras ok")
