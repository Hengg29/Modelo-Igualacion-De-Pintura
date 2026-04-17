"""
Entrenamiento y comparación de modelos de clasificación.

Modelos evaluados:
  1. K-Nearest Neighbors   — baseline natural para colores (distancia perceptual)
  2. SVM (kernel RBF)      — fronteras no lineales en espacio Lab
  3. Random Forest         — ensemble robusto, maneja ruido bien
  4. Regresión Logística   — baseline lineal, referencia de complejidad mínima

Pipeline por modelo:
  StandardScaler → Modelo → Predicción

Métricas:
  - Accuracy (top-1 y top-3)
  - F1-score macro
  - Delta-E medio de las predicciones incorrectas
  - Tiempo de entrenamiento e inferencia
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from skimage.color import deltaE_ciede2000

from dataset.ral_colors import RAL_COLORS

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "dataset")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
RANDOM_SEED = 42


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X_train = train[["L", "a", "b"]].values
    y_train = train["label"].values
    X_test  = test[["L", "a", "b"]].values
    y_test  = test["label"].values
    return X_train, y_train, X_test, y_test


def mean_delta_e_error(y_true, y_pred):
    """Delta-E medio entre el color predicho y el real (solo errores)."""
    errors = []
    for true_lbl, pred_lbl in zip(y_true, y_pred):
        if true_lbl == pred_lbl:
            continue
        true_hex = RAL_COLORS[true_lbl]["hex"]
        pred_hex = RAL_COLORS[pred_lbl]["hex"]
        lab_true = _hex_to_lab(true_hex)
        lab_pred = _hex_to_lab(pred_hex)
        de = deltaE_ciede2000(lab_true.reshape(1,1,3), lab_pred.reshape(1,1,3)).item()
        errors.append(de)
    return float(np.mean(errors)) if errors else 0.0


def _hex_to_lab(hex_str):
    from skimage.color import rgb2lab
    h = hex_str.lstrip("#")
    rgb = np.array([int(h[i:i+2], 16)/255.0 for i in (0,2,4)])
    return rgb2lab(rgb.reshape(1,1,3)).reshape(3)


def top3_accuracy(model, X, y):
    """Porcentaje de muestras donde el label correcto está en las 3 predicciones más probables."""
    proba   = model.predict_proba(X)
    classes = np.array(model.classes_)
    total   = 0
    for j in range(len(y)):
        top3_labels = classes[np.argsort(proba[j])[-3:]]
        if y[j] in top3_labels:
            total += 1
    return total / len(y)


def build_models():
    return {
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)),
        ]),
        "KNN (k=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=3, metric="euclidean", n_jobs=-1)),
        ]),
        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                          probability=True, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=None,
                                              random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000,
                                          solver="lbfgs", random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
    }


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO Y COMPARACIÓN DE MODELOS")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_data()
    print(f"\nDataset cargado:")
    print(f"  Train: {X_train.shape[0]:,} muestras  |  Test: {X_test.shape[0]:,} muestras")
    print(f"  Features: L, a, b  |  Clases: {len(np.unique(y_train))}")

    models  = build_models()
    results = {}

    print("\n" + "-" * 60)
    print(f"{'Modelo':<22} {'Acc':<8} {'Top-3':<8} {'F1':<8} {'ΔE err':<10} {'Tiempo'}")
    print("-" * 60)

    best_name  = None
    best_score = -1

    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        y_pred = pipeline.predict(X_test)
        infer_time = (time.time() - t1) / len(X_test) * 1000  # ms por muestra

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
        top3 = top3_accuracy(pipeline, X_test, y_test)
        de_e = mean_delta_e_error(y_test, y_pred)

        print(f"{name:<22} {acc:.4f}   {top3:.4f}   {f1:.4f}   {de_e:6.2f}      {train_time:.1f}s")

        results[name] = {
            "accuracy":     round(acc,  4),
            "top3_accuracy":round(top3, 4),
            "f1_macro":     round(f1,   4),
            "mean_de_error":round(de_e, 2),
            "train_time_s": round(train_time, 2),
            "infer_ms":     round(infer_time, 3),
        }

        if acc > best_score:
            best_score = acc
            best_name  = name
            best_pipeline = pipeline

    print("-" * 60)
    print(f"\n✓ Mejor modelo: {best_name} (accuracy={best_score:.4f})")

    # Guardar mejor modelo
    os.makedirs(SAVED_DIR, exist_ok=True)
    model_path   = os.path.join(SAVED_DIR, "best_model.pkl")
    meta_path    = os.path.join(SAVED_DIR, "model_meta.json")
    results_path = os.path.join(SAVED_DIR, "results.json")

    joblib.dump(best_pipeline, model_path)

    meta = {
        "best_model":  best_name,
        "accuracy":    round(best_score, 4),
        "n_classes":   int(len(np.unique(y_train))),
        "features":    ["L", "a", "b"],
    }
    with open(meta_path,    "w") as f: json.dump(meta,    f, indent=2)
    with open(results_path, "w") as f: json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado → {model_path}")
    print(f"Resultados      → {results_path}")

    # Reporte detallado del mejor modelo
    print(f"\n{'='*60}")
    print(f"  REPORTE DETALLADO — {best_name}")
    print("=" * 60)
    y_pred_best = best_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_best, zero_division=0, digits=3))


if __name__ == "__main__":
    main()
