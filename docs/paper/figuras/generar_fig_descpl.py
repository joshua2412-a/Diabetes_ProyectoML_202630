"""Diagrama de DES-CPL (Figura 1 del artículo).

Formato de la guía de metodología: PDF vectorial, 6.3 pulgadas de ancho,
Latin Modern Sans de 12 pt, líneas de 2 pt, texto en inglés y sin título
dentro de la figura (el título va en el \\caption).
Uso: python docs/paper/figuras/generar_fig_descpl.py
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
for f in ["lmsans10-regular.otf", "lmsans10-bold.otf"]:
    try: fm.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/lm/" + f)
    except Exception: pass
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
                     "font.size": 12, "pdf.fonttype": 42, "mathtext.fontset": "stixsans"})
AZUL, OCRE, GRIS = "#1F3A5F", "#B7791F", "#555555"
fig, ax = plt.subplots(figsize=(6.3, 6.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 99); ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)  # tamaño final: 6.3 in
W, H = 44, 19
XL, XR = 2, 54                     # columna de entrenamiento y de predicción
YS = [73, 50, 27, 4]               # borde inferior de las cuatro filas
def caja(x, y, titulo, lineas, color, relleno="white", grosor=2):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.25,rounding_size=1.2",
                                lw=grosor, ec=color, fc=relleno))
    ax.text(x + 2, y + H - 4.0, titulo, fontsize=12, weight="bold", color=color, va="center")
    for j, l in enumerate(lineas):
        ax.text(x + 2, y + H - 8.8 - j * 4.4, l, fontsize=12, color="#222222", va="center")
def flecha(p0, p1, conexion="arc3"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=2,
                                 color=GRIS, connectionstyle=conexion, shrinkA=0, shrinkB=0))
ax.text(XL, 96, "Training (inside each outer fold)", fontsize=12, style="italic", color=GRIS)
ax.text(XR, 96, "Prediction for a new patient", fontsize=12, style="italic", color=GRIS)
izq = [("1. Base models", ["7 course models", "tuned with 60 trials"]),
       ("2. Out-of-fold scores", ["K = 3 folds,", "grouped by patient"]),
       ("3. Pruning", ["keep the M' = 4 best", "by out-of-fold AUC-PR"]),
       ("4. Profiles", ["logits of the M' scores", "one k-d tree per class"])]
der = [("5. Local region", ["k nearest positives and", "k nearest negatives;", "same patient excluded"]),
       ("6. Competence", ["class-wise Brier score", "weighted by the local", r"prevalence $\hat{q}(x)$"]),
       ("7. Shrinkage", [r"$(1-\lambda)\,C_m + \lambda\,C^{glob}_m$", r"chosen: $\lambda$ = 0.9 to 1"]),
       ("8. Weights and mix", [r"$w_m \propto \exp(-C_m/\tau)$", r"$\hat{f}(x)=\Sigma_m\, w_m f_m(x)$"])]
for y, (t, l) in zip(YS, izq): caja(XL, y, t, l, AZUL)
for i, (y, (t, l)) in enumerate(zip(YS, der)):
    caja(XR, y, t, l, OCRE, "#FFF6E3" if i == 2 else "white", 3 if i == 2 else 2)
for k in range(3):
    flecha((XL + W / 2, YS[k] - 0.4), (XL + W / 2, YS[k + 1] + H + 0.4))
    flecha((XR + W / 2, YS[k] - 0.4), (XR + W / 2, YS[k + 1] + H + 0.4))
# De los perfiles (abajo a la izquierda) a la región local (arriba a la derecha).
xm = (XL + W + XR) / 2
ax.plot([XL + W + 0.4, xm, xm], [YS[3] + H / 2, YS[3] + H / 2, YS[0] + H / 2], color=GRIS, lw=2)
flecha((xm, YS[0] + H / 2), (XR - 0.4, YS[0] + H / 2))
salida = Path(__file__).resolve().parent / "fig_descpl_diagrama.pdf"
fig.savefig(salida, bbox_inches=None)
