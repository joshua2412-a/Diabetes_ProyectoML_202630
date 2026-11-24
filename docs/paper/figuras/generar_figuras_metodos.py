"""Figuras de la sección de Materiales y métodos del artículo.

- fig_eda.pdf: (a) similitud entre encuentros del mismo paciente frente a su
  referencia por permutación; (b) AUC univariado de los mejores predictores
  frente al techo del oráculo y al desempeño de los modelos.
- fig_validacion.pdf: esquema de la partición por paciente y de la
  validación cruzada anidada (5 pliegues externos, 3 internos).

Lee solo archivos de results/ (no entrena nada). Mismo formato que el resto
de figuras: PDF vectorial, 6.3 pulgadas, Latin Modern Sans de 12 pt, líneas
de 2 pt, texto en inglés y sin título dentro de la figura.
Uso (raíz del repositorio): python docs/paper/figuras/generar_figuras_metodos.py
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch

RAIZ = Path(__file__).resolve().parents[3]
R, O = RAIZ / "results", Path(__file__).resolve().parent
for f in ["lmsans10-regular.otf", "lmsans10-bold.otf", "lmsans10-oblique.otf"]:
    try: fm.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/lm/" + f)
    except Exception: pass
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3,
                     "pdf.fonttype": 42, "mathtext.fontset": "stixsans", "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})
W = 6.3
AZUL, OCRE, GRIS = "#1F3A5F", "#B7791F", "#777777"

# ---------- Figura EDA ----------
sim = pd.read_csv(R / "similitud_intrapaciente.csv").set_index("variable")
VARS = [("age_ord", "Age group"), ("gender", "Sex"), ("race", "Race"),
        ("especialidad", "Admitting specialty"), ("discharge_disposition_id", "Discharge disposition"),
        ("number_inpatient", "Prior inpatient visits"), ("diag_1_grupo", "Primary diagnosis group"),
        ("readmitted_30", "Label: readmitted < 30 d")]
rk = pd.read_csv(R / "ranking_predictores.csv").sort_values("AUC univariado", ascending=False).head(8)
NOMBRES = {"number_inpatient": "Prior inpatient visits", "hospitalizacion_previa": "Any prior inpatient visit",
           "discharge_disposition_id": "Discharge disposition", "time_in_hospital": "Length of stay",
           "number_diagnoses": "Number of diagnoses", "num_medications": "Number of medications",
           "insulin": "Insulin", "number_emergency": "Prior emergency visits",
           "diag_1_grupo": "Primary diagnosis group", "num_lab_procedures": "Lab procedures"}
techo = pd.read_csv(R / "techo_desempeno.csv")
oraculo = float(techo.loc[techo["predictores usados"] == 3, "AUC del oráculo (fuera de muestra)"].iloc[0])

fig, (a, b) = plt.subplots(2, 1, figsize=(W, 6.6), gridspec_kw={"height_ratios": [1, 1]})
for i, (v, nombre) in enumerate(VARS[::-1]):
    o, p = sim.loc[v, "observado"], sim.loc[v, "permutado"]
    es_etiqueta = v == "readmitted_30"
    a.plot([p, o], [i, i], color=GRIS, lw=2, zorder=1)
    a.scatter(p, i, s=55, facecolor="white", edgecolor=GRIS, lw=2, zorder=2, marker="o")
    a.scatter(o, i, s=60, color=OCRE if es_etiqueta else AZUL, zorder=3, marker="D" if es_etiqueta else "s")
a.set_yticks(range(len(VARS)), [n for _, n in VARS[::-1]])
a.set_xlim(0, 1.03); a.set_xlabel(r"Within-patient similarity ($\eta^2$ or agreement)")
a.scatter([], [], s=60, color=AZUL, marker="s", label="observed")
a.scatter([], [], s=55, facecolor="white", edgecolor=GRIS, lw=2, label="permuted patients")
a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
a.grid(axis="y", visible=False)
ys = range(len(rk))[::-1]
a2 = b
a2.barh(list(ys), rk["AUC univariado"] - 0.5, left=0.5, height=0.6, color="#C9D6E8", edgecolor=AZUL, lw=1.2)
a2.set_yticks(list(ys), [NOMBRES.get(v, v) for v in rk["variable"]])
a2.axvline(0.5, color=GRIS, ls=":", lw=1.5)
a2.axvline(oraculo, color=OCRE, ls="--", lw=2)
a2.axvline(0.666, color=AZUL, ls="-.", lw=2)
a2.text(oraculo - 0.003, len(rk) - 3.0, f"oracle, top-3\nprofiles ({oraculo:.3f})", color=OCRE, fontsize=12, va="top", ha="right")
a2.text(0.666 + 0.002, 1.2, "best models,\ntest (0.666)", color=AZUL, fontsize=12, va="center")
a2.set_xlim(0.49, 0.72); a2.set_xlabel("Univariate AUC-ROC (training set)")
a2.grid(axis="y", visible=False)
a.annotate("not shared within\npatients", xy=(0.36, 0), xytext=(0.47, 0.75), fontsize=12, color=OCRE,
           arrowprops=dict(arrowstyle="-|>", color=OCRE, lw=1.5))
fig.set_constrained_layout(True)
fig.canvas.draw()
fig.set_layout_engine("none")
for ax, l in zip((a, a2), "ab"):
    caja = ax.get_position()
    fig.text(0.012, min(caja.y1 + 0.006, 0.985), f"({l})", fontsize=13, weight="bold", va="bottom")
fig.savefig(O / "fig_eda.pdf"); plt.close(fig)

# ---------- Figura de validación ----------
plt.rcParams.update({"axes.grid": False})
fig, ax = plt.subplots(figsize=(W, 6.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
C_TR, C_EV, C_IN, C_TE = "#C9D6E8", "#F3D9A6", "#CFE6D2", "#E2E2E2"
def barra(x0, x1, y, h, partes, resaltar, color_r, hatch_r):
    ancho = (x1 - x0) / partes
    for j in range(partes):
        r = j == resaltar
        ax.add_patch(Rectangle((x0 + j * ancho, y), ancho, h, fc=color_r if r else C_TR,
                               ec=AZUL, lw=1.2, hatch=hatch_r if r else None))
def flecha(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=2, color=GRIS))
ax.text(2, 96.5, "1. Split by patient (seed 42), intersection of patients = 0", fontsize=12, style="italic", color=GRIS)
ax.add_patch(Rectangle((2, 87), 78, 6, fc=C_TR, ec=AZUL, lw=1.2))
ax.add_patch(Rectangle((80, 87), 18, 6, fc=C_TE, ec=GRIS, lw=1.2, hatch="xx"))
ax.text(41, 90, "Training: 79,473 encounters", ha="center", va="center", fontsize=12)
ax.text(89, 90, "Test", ha="center", va="center", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
flecha((41, 86.5), (41, 82.5))
ax.text(2, 80, "2. Outer loop: 5 folds, StratifiedGroupKFold", fontsize=12, style="italic", color=GRIS)
for k in range(5):
    y = 72 - 6.3 * k
    ax.text(2, y + 2.2, f"Fold {k + 1}", fontsize=12, va="center")
    barra(16, 80, y, 4.4, 5, k, C_EV, "//")
flecha((48, 46.3), (48, 42.5))
ax.text(2, 40, "3. Inner loop, inside each outer training set: 3 folds", fontsize=12, style="italic", color=GRIS)
for k in range(3):
    y = 32.5 - 5.6 * k
    ax.text(2, y + 2.0, f"Inner {k + 1}", fontsize=12, va="center")
    barra(16, 64, y, 4.0, 3, k, C_IN, "..")
ax.text(67, 34.4, "Base models: tune", fontsize=12, va="center")
ax.text(67, 30.4, "with 60 trials", fontsize=12, va="center")
ax.text(67, 25.0, "DES-CPL: out-of-fold", fontsize=12, va="center")
ax.text(67, 21.0, "scores = selection set;", fontsize=12, va="center")
ax.text(67, 17.0, r"tune $k,\tau,\lambda,M'$ (60)", fontsize=12, va="center")
ax.text(2, 11.0, "4. Refit on the outer training set, score its held-out fold;", fontsize=12, color="#222222")
ax.text(2, 6.6, "    finally refit on all training and score the test set once.", fontsize=12, color="#222222")
ax.legend(handles=[Patch(fc=C_TR, ec=AZUL, label="training"),
                   Patch(fc=C_EV, ec=AZUL, hatch="//", label="outer held-out"),
                   Patch(fc=C_IN, ec=AZUL, hatch="..", label="inner held-out"),
                   Patch(fc=C_TE, ec=GRIS, hatch="xx", label="test")],
          loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=12, handlelength=1.4,
          columnspacing=0.9, handletextpad=0.4)
fig.savefig(O / "fig_validacion.pdf", bbox_inches=None); plt.close(fig)
print("listo:", O / "fig_eda.pdf", O / "fig_validacion.pdf")
