"""Colour arithmetic for the page's palette: sRGB, CIELAB, OKLab and OKLCH, WCAG contrast,
and colour vision deficiency simulation.

Simulation follows Machado, Oliveira and Fernandes (2009), "A Physiologically-based Model
for Simulation of Color Vision Deficiency", at full severity (1.0), applied to linear RGB.
The matrices are the paper's published severity 1.0 values.

Everything here works on numpy arrays of shape (..., 3) with sRGB in [0, 1].
"""

from __future__ import annotations

import numpy as np

MACHADO_2009 = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
}


def to_linear(srgb: np.ndarray) -> np.ndarray:
    c = np.asarray(srgb, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(linear, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(srgb: np.ndarray, kind: str) -> np.ndarray:
    """sRGB as seen with a full dichromacy, clipped back into sRGB."""
    if kind == "normal":
        return np.asarray(srgb, np.float64)
    lin = to_linear(srgb)
    return to_srgb(lin @ MACHADO_2009[kind].T)


def relative_luminance(srgb: np.ndarray) -> np.ndarray:
    return to_linear(srgb) @ np.array([0.2126, 0.7152, 0.0722])


def contrast_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = np.maximum(la, lb), np.minimum(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def lab(srgb: np.ndarray) -> np.ndarray:
    """CIELAB, D65."""
    lin = to_linear(srgb)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1
    )


def delta_e76(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab(a) - lab(b), axis=-1)


def oklab_from_srgb(srgb: np.ndarray) -> np.ndarray:
    lin = to_linear(srgb)
    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return np.cbrt(lin @ m1.T) @ m2.T


def srgb_from_oklch(lightness, chroma, hue_deg) -> tuple[np.ndarray, np.ndarray]:
    """sRGB for OKLCH, and whether each colour is inside the sRGB gamut before clipping."""
    lightness, chroma, hue_deg = np.broadcast_arrays(
        np.asarray(lightness, np.float64),
        np.asarray(chroma, np.float64),
        np.asarray(hue_deg, np.float64),
    )
    h = np.radians(hue_deg)
    lab_ = np.stack([lightness, chroma * np.cos(h), chroma * np.sin(h)], -1)
    m2i = np.array(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ]
    )
    m1i = np.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )
    lin = (lab_ @ m2i.T) ** 3 @ m1i.T
    inside = np.all((lin >= -1e-6) & (lin <= 1 + 1e-6), axis=-1)
    return to_srgb(lin), inside


def max_chroma_in_gamut(lightness: float, hue_deg: float, step: float = 0.002) -> float:
    c = 0.0
    while c < 0.4:
        _, ok = srgb_from_oklch(lightness, c + step, hue_deg)
        if not bool(ok):
            break
        c += step
    return round(c, 4)
