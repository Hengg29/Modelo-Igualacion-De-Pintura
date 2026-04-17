"""
Evaluación visual del mejor modelo.

Genera:
  1. Gráfica de barras — comparación de accuracy entre modelos
  2. Matriz de confusión (top-20 clases más confundidas)
  3. Distribución de Delta-E de errores
  4. Curva de accuracy vs k (para KNN)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.color import deltaE_ciede2000
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset.ral_colors import RAL_COLORS

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "dataset")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _hex_to_lab(hex_str):
    from skimage.color import rgb2lab
    h = hex_str.lstrip("#")
    rgb = np.array([int(h[i:i+2], 16)/255.0 for i in (0,2,4)])
    return rgb2lab(rgb.reshape(1,1,3)).reshape(3)


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return (train[["L","a","b"]].values, train["label"].values,
            test[["L","a","b"]].values,  test["label"].values)


def plot_model_comparison():
    results_path = os.path.join(SAVED_DIR, "results.json")
    if not os.path.exists(results_path):
        print("Primero ejecuta models/train.py"); return

    with open(results_path) as f:
        results = json.load(f)

    names   = list(results.keys())
    acc     = [results[n]["accuracy"]      for n in names]
    top3    = [results[n]["top3_accuracy"] for n in names]
    f1      = [results[n]["f1_macro"]      for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f0f18")
    ax.set_facecolor("#0f0f18")

    bars1 = ax.bar(x - w, acc,  w, label="Accuracy top-1", color="#7c3aed", alpha=.9)
    bars2 = ax.bar(x,     top3, w, label="Accuracy top-3", color="#ec4899", alpha=.9)
    bars3 = ax.bar(x + w, f1,   w, label="F1 macro",       color="#f59e0b", alpha=.9)

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + .005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#e0e0f0")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", color="#c0c0e0", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Métrica", color="#c0c0e0")
    ax.set_title("Comparación de modelos — Clasificación de colores RAL", color="#f0f0ff", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ddd", fontsize=8)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    ax.yaxis.label.set_color("#c0c0e0")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Guardado: {path}")


def plot_delta_e_distribution():
    X_train, y_train, X_test, y_test = load_data()
    model = joblib.load(os.path.join(SAVED_DIR, "best_model.pkl"))
    y_pred = model.predict(X_test)

    de_errors = []
    de_correct = []
    for true_lbl, pred_lbl in zip(y_test, y_pred):
        lab_t = _hex_to_lab(RAL_COLORS[true_lbl]["hex"])
        lab_p = _hex_to_lab(RAL_COLORS[pred_lbl]["hex"])
        de = deltaE_ciede2000(lab_t.reshape(1,1,3), lab_p.reshape(1,1,3)).item()
        if true_lbl == pred_lbl:
            de_correct.append(de)
        else:
            de_errors.append(de)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f0f18")
    ax.set_facecolor("#0f0f18")

    ax.hist(de_errors,  bins=40, color="#f87171", alpha=.8, label=f"Errores (n={len(de_errors)})")
    ax.axvline(1,  color="#facc15", linestyle="--", linewidth=1.2, label="ΔE=1 (imperceptible)")
    ax.axvline(5,  color="#fb923c", linestyle="--", linewidth=1.2, label="ΔE=5 (notable)")

    ax.set_xlabel("Delta-E 2000", color="#c0c0e0")
    ax.set_ylabel("Frecuencia",   color="#c0c0e0")
    ax.set_title("Distribución de Delta-E en predicciones incorrectas", color="#f0f0ff", fontsize=11, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ddd", fontsize=8)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")

    mean_de = np.mean(de_errors) if de_errors else 0
    ax.text(0.98, 0.95, f"ΔE medio error: {mean_de:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            color="#fbbf24", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#555"))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "delta_e_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Guardado: {path}")


def plot_knn_k_curve():
    """Accuracy vs k para KNN — justifica la elección de k."""
    X_train, y_train, X_test, y_test = load_data()
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    ks  = list(range(1, 21))
    acc = []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        knn.fit(Xtr, y_train)
        acc.append(accuracy_score(y_test, knn.predict(Xte)))

    best_k   = ks[int(np.argmax(acc))]
    best_acc = max(acc)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f0f18")
    ax.set_facecolor("#0f0f18")

    ax.plot(ks, acc, color="#7c6ff7", linewidth=2, marker="o", markersize=5)
    ax.axvline(best_k, color="#ec4899", linestyle="--", linewidth=1.5,
               label=f"Mejor k={best_k} (acc={best_acc:.4f})")
    ax.fill_between(ks, acc, alpha=0.1, color="#7c6ff7")

    ax.set_xlabel("k (número de vecinos)", color="#c0c0e0")
    ax.set_ylabel("Accuracy",              color="#c0c0e0")
    ax.set_title("KNN — Accuracy vs k en test set", color="#f0f0ff", fontsize=11, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ddd", fontsize=8)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "knn_k_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Guardado: {path}")


def print_summary():
    results_path = os.path.join(SAVED_DIR, "results.json")
    if not os.path.exists(results_path):
        return
    with open(results_path) as f:
        results = json.load(f)
    print("\n" + "="*60)
    print("  RESUMEN DE MÉTRICAS")
    print("="*60)
    print(f"{'Modelo':<22} {'Acc':<8} {'Top3':<8} {'F1':<8} {'ΔE err'}")
    print("-"*60)
    for name, r in results.items():
        print(f"{name:<22} {r['accuracy']:<8.4f} {r['top3_accuracy']:<8.4f} "
              f"{r['f1_macro']:<8.4f} {r['mean_de_error']:.2f}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Generando gráficas de evaluación...")
    plot_model_comparison()
    plot_delta_e_distribution()
    plot_knn_k_curve()
    print_summary()
    print(f"\nTodas las gráficas guardadas en: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
