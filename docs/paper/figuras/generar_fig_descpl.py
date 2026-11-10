import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Latin Modern Sans","DejaVu Sans"],
 "font.size":12,"pdf.fonttype":42,"savefig.bbox":"tight","mathtext.fontset":"dejavusans"})
fig,ax=plt.subplots(figsize=(6.3,4.1));ax.set_xlim(-0.5,102);ax.set_ylim(0,66);ax.axis("off")
C1,C2,C3="#1F3A5F","#2A7F8E","#B7791F"
W,G=22.5,3.3
xs=[0.8+i*(W+G) for i in range(4)]
def box(x,y,h,title,lines,col,fill="#FFFFFF"):
    ax.add_patch(FancyBboxPatch((x,y),W,h,boxstyle="round,pad=0.3,rounding_size=1.5",lw=2,ec=col,fc=fill))
    ax.text(x+W/2,y+h-3.6,title,ha="center",va="center",fontsize=9.8,weight="bold",color=col)
    for j,l in enumerate(lines):
        ax.text(x+W/2,y+h-9.6-j*4.6,l,ha="center",va="center",fontsize=9,color="#222222")
def arrow(x0,y0,x1,y1,col="#555555",ls="-"):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=13,lw=2,color=col,linestyle=ls,shrinkA=1,shrinkB=1))
H=22
ax.text(0.8,63.5,"Entrenamiento (dentro de cada pliegue externo)",fontsize=10,color="#555555",style="italic")
top=[("Modelos base",["$M=7$ modelos","ajustados","$f_1,\\dots,f_M$"]),
     ("Fuera de pliegue",["predicciones con","$K$ pliegues por","paciente $\\tilde f_m(x_i)$"]),
     ("Selección",["perfiles $\\phi(x_i)$,","etiquetas $y_i$,","pacientes $g_i$"]),
     ("Dos índices",["uno por clase:","positivos y","negativos"])]
for x,(t,l) in zip(xs,top): box(x,37,H,t,l,C1)
bot=[("Consulta",["perfil $\\phi(x)$","y $f_m(x)$","para cada $m$"],C2,"#FFFFFF"),
     ("Región local",["$k$ vecinos","positivos y $k$","negativos; sin el","mismo paciente"],C2,"#FFFFFF"),
     ("Competencia",["Brier local","balanceado","por clase $C_m(x)$"],C3,"#FFF8E8"),
     ("Combinación",["pesos softmax","$w_m\\propto e^{-C_m/\\tau}$","$\\hat f=\\Sigma_m\\, w_m f_m$"],C3,"#FFF8E8")]
for x,(t,l,c,f) in zip(xs,bot): box(x,3,H+3,t,l,c,f)
ax.text(0.8,31.5,"Predicción para un paciente nuevo $x$",fontsize=10,color="#555555",style="italic")
for i in range(3):
    arrow(xs[i]+W+0.4,48,xs[i+1]-0.4,48); arrow(xs[i]+W+0.4,15.5,xs[i+1]-0.4,15.5)
arrow(xs[3]+W/2,36.5,xs[1]+W/2,28.5,col="#888888",ls="--")
fig.savefig("fig_descpl_diagrama.pdf");fig.savefig("preview.png",dpi=110)
