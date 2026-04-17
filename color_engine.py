import re
import os

import numpy as np
import joblib
from skimage.color import rgb2lab, deltaE_ciede2000

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved", "best_model.pkl")
_model = None


def _get_model():
    global _model
    if _model is None and os.path.exists(_MODEL_PATH):
        _model = joblib.load(_MODEL_PATH)
    return _model


# ── color math ────────────────────────────────────────────────────

def _normalize_hex(h: str) -> str:
    h = h.strip().upper().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9A-F]{6}", h):
        raise ValueError(f"Hex inválido: {h}")
    return f"#{h}"


def _hex_to_lab(hex_str: str) -> np.ndarray:
    h = _normalize_hex(hex_str).lstrip("#")
    rgb = np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    return rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)


def _interpret(de: float) -> str:
    if de < 1.0:  return "Imperceptible — prácticamente idénticos al ojo humano."
    if de < 2.0:  return "Muy cercano — diferencia perceptible solo por un experto."
    if de < 3.5:  return "Ligera diferencia — notada en observación directa."
    if de < 5.0:  return "Diferencia moderada — visible pero aceptable."
    if de < 10.0: return "Diferencia notable — colores claramente distintos."
    return "Diferencia grande — los colores no coinciden."


# ── public API ────────────────────────────────────────────────────

def compare_hex_colors(hex_a: str, hex_b: str) -> dict:
    """Compara dos colores hex con Delta-E 2000."""
    lab_a = _hex_to_lab(hex_a)
    lab_b = _hex_to_lab(hex_b)
    de = float(deltaE_ciede2000(lab_a.reshape(1,1,3), lab_b.reshape(1,1,3)).item())
    similarity_pct = max(0.0, min(100.0, 100.0 * (1.0 - de / 50.0)))
    return {
        "delta_e":        round(de, 2),
        "similarity_pct": round(similarity_pct, 1),
        "interpretation": _interpret(de),
        "hex_a":          _normalize_hex(hex_a),
        "hex_b":          _normalize_hex(hex_b),
    }


def classify_color(hex_str: str) -> dict:
    """
    Usa el modelo entrenado para clasificar un color en la paleta RAL.
    Devuelve el código RAL más probable + top-3 con probabilidades.
    """
    model = _get_model()
    if model is None:
        return {"error": "Modelo no entrenado. Ejecuta: python models/train.py"}

    lab = _hex_to_lab(hex_str)
    X = lab.reshape(1, -1)

    prediction = model.predict(X)[0]
    probas = model.predict_proba(X)[0]
    classes = model.classes_

    top3_idx = np.argsort(probas)[-3:][::-1]
    top3 = [
        {
            "code":        classes[i],
            "probability": round(float(probas[i]), 4),
        }
        for i in top3_idx
    ]

    return {
        "predicted_code": prediction,
        "top3":           top3,
        "input_hex":      _normalize_hex(hex_str),
    }
