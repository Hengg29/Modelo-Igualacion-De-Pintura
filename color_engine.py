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
    Clasifica un color en la paleta RAL usando Delta-E 2000 exacto.
    Devuelve el código RAL más cercano + top-3 con similitud normalizada.
    """
    from dataset.ral_colors import RAL_COLORS

    lab_input = _hex_to_lab(hex_str)

    distances = []
    for code, info in RAL_COLORS.items():
        lab_ral = _hex_to_lab(info["hex"])
        de = float(deltaE_ciede2000(
            lab_input.reshape(1, 1, 3),
            lab_ral.reshape(1, 1, 3)
        ).item())
        distances.append((code, de))

    distances.sort(key=lambda x: x[1])
    top3_raw = distances[:3]

    max_de = max(d for _, d in top3_raw) or 1.0
    top3 = [
        {
            "code":        code,
            "probability": round(max(0.0, 1.0 - (de / (max_de + 1e-9))), 4),
        }
        for code, de in top3_raw
    ]
    top3[0]["probability"] = max(top3[0]["probability"], 0.9999)

    return {
        "predicted_code": distances[0][0],
        "top3":           top3,
        "input_hex":      _normalize_hex(hex_str),
    }
