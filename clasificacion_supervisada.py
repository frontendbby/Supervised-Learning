# =============================================================
# Clasificación Supervisada - Comparación de Algoritmos
# Dataset: Digits (clasificación binaria: dígito 3 vs resto)
# Algoritmos: Logistic Regression, KNN, SVM, Naive Bayes,
#             Decision Tree, Random Forest
# =============================================================

# ── Importación de librerías ──────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ── Carga y preparación del dataset ──────────────────────────
data = load_digits()
X = data.data
y = (data.target == 3).astype(int)   # Binario: 1 = dígito 3, 0 = resto

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Muestras de entrenamiento : {X_train.shape[0]}")
print(f"Muestras de prueba        : {X_test.shape[0]}")
print(f"Proporción de clase 1     : {y.mean():.2%}\n")

# ── Definición de modelos ─────────────────────────────────────
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=5)"          : KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)"          : SVC(kernel="rbf", probability=True, random_state=42),
    "Naive Bayes"        : GaussianNB(),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
}

colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

# ── Entrenamiento, evaluación y recolección de métricas ───────
resultados = {}

print("=" * 60)
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    cv_scores   = cross_val_score(modelo, X, y, cv=5, scoring="roc_auc")

    reporte = classification_report(y_test, y_pred, output_dict=True)

    resultados[nombre] = {
        "modelo"   : modelo,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "fpr"      : fpr,
        "tpr"      : tpr,
        "auc"      : roc_auc,
        "cv_auc"   : cv_scores.mean(),
        "precision": reporte["weighted avg"]["precision"],
        "recall"   : reporte["weighted avg"]["recall"],
        "f1"       : reporte["weighted avg"]["f1-score"],
        "accuracy" : reporte["accuracy"],
    }

    print(f"\n{'─'*50}")
    print(f"  {nombre}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Otro", "Dígito 3"]))
    print(f"  ROC AUC (test)     : {roc_auc:.4f}")
    print(f"  ROC AUC (CV 5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("=" * 60)

# ── Dashboard de visualizaciones ─────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0F1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax_roc  = fig.add_subplot(gs[0, :2])   # Curvas ROC (grande)
ax_bar  = fig.add_subplot(gs[0, 2])    # Barras AUC
ax_cms  = [fig.add_subplot(gs[1, i]) for i in range(3)]   # 3 matrices de confusión
ax_cms2 = [fig.add_subplot(gs[2, i]) for i in range(3)]   # 3 matrices restantes
ax_conf = ax_cms + ax_cms2

STYLE = dict(facecolor="#1A1D27", edgecolor="#333", linewidth=0.8)
TEXT  = "#E0E0E0"
GRID  = "#2A2D3A"

# ── Panel 1: Curvas ROC ───────────────────────────────────────
ax_roc.set_facecolor("#1A1D27")
for (nombre, res), color in zip(resultados.items(), colores):
    ax_roc.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{nombre}  (AUC={res['auc']:.3f})")
ax_roc.plot([0,1],[0,1], "w--", lw=1, alpha=0.4)
ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
ax_roc.set_xlabel("Tasa de Falsos Positivos", color=TEXT, fontsize=11)
ax_roc.set_ylabel("Tasa de Verdaderos Positivos", color=TEXT, fontsize=11)
ax_roc.set_title("Curvas ROC — Todos los modelos", color=TEXT, fontsize=13, fontweight="bold", pad=12)
ax_roc.legend(loc="lower right", fontsize=9, framealpha=0.2,
              labelcolor=TEXT, facecolor="#1A1D27")
ax_roc.tick_params(colors=TEXT)
ax_roc.grid(True, color=GRID, linewidth=0.5)
for spine in ax_roc.spines.values():
    spine.set_edgecolor(GRID)

# ── Panel 2: Barras AUC ───────────────────────────────────────
nombres = list(resultados.keys())
aucs    = [resultados[n]["auc"] for n in nombres]
cv_aucs = [resultados[n]["cv_auc"] for n in nombres]
x_pos   = np.arange(len(nombres))

ax_bar.set_facecolor("#1A1D27")
bars = ax_bar.bar(x_pos, aucs, color=colores, width=0.6, alpha=0.85, zorder=3)
ax_bar.scatter(x_pos, cv_aucs, color="white", s=50, zorder=5,
               label="CV AUC (5-fold)", edgecolors="#555", linewidth=0.8)
for bar, val in zip(bars, aucs):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=8)
ax_bar.set_xticks(x_pos)
short = ["LogReg", "KNN", "SVM", "NB", "DTree", "RF"]
ax_bar.set_xticklabels(short, color=TEXT, fontsize=9)
ax_bar.set_ylim([0.9, 1.01])
ax_bar.set_title("AUC por Modelo", color=TEXT, fontsize=11, fontweight="bold")
ax_bar.set_ylabel("AUC", color=TEXT, fontsize=10)
ax_bar.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor="#1A1D27")
ax_bar.tick_params(colors=TEXT)
ax_bar.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
for spine in ax_bar.spines.values():
    spine.set_edgecolor(GRID)

# ── Paneles 3-8: Matrices de confusión ───────────────────────
for ax, (nombre, res), color in zip(ax_conf, resultados.items(), colores):
    cm   = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Otro", "Díg.3"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_facecolor("#1A1D27")
    ax.set_title(nombre, color=TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("Predicho", color=TEXT, fontsize=8)
    ax.set_ylabel("Real", color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    # Texto de celdas en blanco para contraste
    for text in ax.texts:
        text.set_color("white")

fig.suptitle("Comparación de Algoritmos de Aprendizaje Supervisado\nDigits Dataset — Clasificación Binaria (Dígito 3 vs Resto)",
             color=TEXT, fontsize=14, fontweight="bold", y=0.98)

plt.savefig("/mnt/user-data/outputs/dashboard_clasificacion.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\nDashboard guardado en: dashboard_clasificacion.png")

# ── Resumen final ─────────────────────────────────────────────
print("\n" + "="*60)
print("  RESUMEN COMPARATIVO")
print("="*60)
print(f"{'Modelo':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-"*67)
for nombre, res in resultados.items():
    print(f"{nombre:<22} {res['accuracy']:>9.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>8.4f} {res['f1']:>8.4f} {res['auc']:>8.4f}")
print("="*60)

mejor = max(resultados, key=lambda n: resultados[n]["auc"])
print(f"\n🏆 Mejor modelo por AUC: {mejor} ({resultados[mejor]['auc']:.4f})")
