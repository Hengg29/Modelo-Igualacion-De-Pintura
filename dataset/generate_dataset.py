"""
Genera el dataset sintético de entrenamiento y prueba.

Por cada color RAL se crean N muestras simulando variaciones reales:
  - Ruido de sensor de cámara (Gaussiano)
  - Variación de exposición (brillo)
  - Desplazamiento de temperatura de color (luz cálida / fría)
  - Variación de saturación

Esto permite entrenar un clasificador robusto a condiciones de iluminación,
que es exactamente el reto real al comparar pinturas con una cámara.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from skimage.color import rgb2lab

from dataset.ral_colors import RAL_COLORS

RANDOM_SEED   = 42
N_TRAIN       = 150   # muestras de entrenamiento por color
N_TEST        = 30    # muestras de prueba por color
NOISE_SIGMA   = 4.0   # ruido Lab (unidades Lab)
EXPOSURE_STD  = 0.12  # variación de exposición (fracción de L)
TEMP_STD      = 3.5   # desplazamiento de temperatura (unidades a/b)
SAT_STD       = 0.08  # variación de saturación (fracción)


def hex_to_rgb01(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])


def rgb01_to_lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)


def augment_lab(lab: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Genera n variaciones de un color Lab simulando condiciones reales."""
    samples = np.tile(lab, (n, 1)).astype(np.float64)

    # 1. Ruido de sensor (Gaussiano en Lab)
    samples += rng.normal(0, NOISE_SIGMA, samples.shape)

    # 2. Variación de exposición (afecta principalmente L*)
    exposure = rng.normal(0, EXPOSURE_STD * lab[0], (n, 1))
    samples[:, 0:1] += exposure

    # 3. Temperatura de color: desplaza a* y b*
    # Luz cálida: +a, +b  /  Luz fría: -a, -b
    temp_shift = rng.normal(0, TEMP_STD, (n, 1))
    samples[:, 1:2] += temp_shift * 0.8   # a*
    samples[:, 2:3] += temp_shift * 1.2   # b*

    # 4. Variación de saturación (escala a* y b* alrededor del neutro)
    sat_factor = 1.0 + rng.normal(0, SAT_STD, (n, 1))
    samples[:, 1:3] *= np.clip(sat_factor, 0.5, 1.8)

    # Clamp a rangos válidos de Lab
    samples[:, 0] = np.clip(samples[:, 0], 0, 100)
    samples[:, 1] = np.clip(samples[:, 1], -128, 127)
    samples[:, 2] = np.clip(samples[:, 2], -128, 127)

    return samples


def build_split(n_per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for code, info in RAL_COLORS.items():
        rgb = hex_to_rgb01(info["hex"])
        lab = rgb01_to_lab(rgb)
        augmented = augment_lab(lab, n_per_class, rng)
        for sample in augmented:
            rows.append({
                "L": round(sample[0], 4),
                "a": round(sample[1], 4),
                "b": round(sample[2], 4),
                "label": code,
                "name":  info["name"],
                "hex":   info["hex"],
            })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    out = os.path.join(os.path.dirname(__file__))

    print(f"Generando dataset — {len(RAL_COLORS)} colores RAL")
    print(f"  Train: {N_TRAIN} muestras/color = {N_TRAIN * len(RAL_COLORS):,} total")
    print(f"  Test:  {N_TEST}  muestras/color = {N_TEST  * len(RAL_COLORS):,} total")

    train_df = build_split(N_TRAIN, rng)
    test_df  = build_split(N_TEST,  rng)

    train_path = os.path.join(out, "train.csv")
    test_path  = os.path.join(out, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"\n✓ train.csv guardado → {len(train_df):,} filas")
    print(f"✓ test.csv  guardado → {len(test_df):,} filas")
    print(f"\nVariables de entrada: L, a, b  |  Variable objetivo: label (código RAL)")
    print(f"Clases únicas: {train_df['label'].nunique()}")


if __name__ == "__main__":
    main()
