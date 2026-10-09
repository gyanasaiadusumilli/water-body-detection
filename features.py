"""Spectral bands and water indices shared by the ML models and the CNN."""
import numpy as np

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
INDICES = ["ndwi", "mndwi", "ndvi", "awei_sh"]
FEATURES = BANDS + INDICES
IGNORE = 255
EPS = 1e-6


def compute_indices(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Water and vegetation indices from the 6 reflectance bands.

    x: array with the 6 bands along `axis`. Returns the 4 indices stacked along `axis`.
      NDWI   (McFeeters 1996): (green - nir) / (green + nir)
      MNDWI  (Xu 2006):       (green - swir1) / (green + swir1), better in built-up areas
      NDVI:                   (nir - red) / (nir + red), separates vegetation from water
      AWEI_sh (Feyisa 2014):  blue + 2.5*green - 1.5*(nir + swir1) - 0.25*swir2, handles shadows
    """
    blue, green, red, nir, swir1, swir2 = (np.take(x, i, axis=axis) for i in range(6))
    ndwi = (green - nir) / (green + nir + EPS)
    mndwi = (green - swir1) / (green + swir1 + EPS)
    ndvi = (nir - red) / (nir + red + EPS)
    # Near-zero or slightly negative reflectance (common after Landsat scaling) makes the
    # ratios explode; clip to their valid range so outliers don't distort the models.
    ndwi, mndwi, ndvi = (np.clip(v, -1.0, 1.0) for v in (ndwi, mndwi, ndvi))
    awei = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    return np.stack([ndwi, mndwi, ndvi, awei], axis=axis).astype(np.float32)


def add_indices(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """6 bands -> 10 channels (bands + indices)."""
    x = x.astype(np.float32)
    return np.concatenate([x, compute_indices(x, axis)], axis=axis)


def tiles_to_pixels(X: np.ndarray, y: np.ndarray):
    """(N, 6, H, W) tiles + (N, H, W) labels -> (P, 10) features and (P,) labels, ignore pixels dropped."""
    feats = add_indices(X, axis=1)                        # (N, 10, H, W)
    feats = feats.transpose(0, 2, 3, 1).reshape(-1, feats.shape[1])
    labels = y.reshape(-1)
    valid = labels != IGNORE
    return feats[valid], labels[valid].astype(np.int64)
