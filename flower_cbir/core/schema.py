"""Fixed, report-defined feature order. Shared by queries, CSV and PostgreSQL."""
import numpy as np

EXTRACTOR_VERSION = 'report1-163-v3'
GROUP_LENGTHS = (78, 18, 28, 39)
GROUP_SLICES = (slice(0, 78), slice(78, 96), slice(96, 124), slice(124, 163))
F1_NAMES = [f'hs_h{h:02d}_s{s:02d}' for h in range(12) for s in range(6)] + [f'v_{v:02d}' for v in range(6)]
F2_NAMES = [f'lab_r{r}_{channel}_{stat}' for r in range(1, 4) for channel in ('L', 'a', 'b') for stat in ('mean', 'std')]
F3_NAMES = [f'lbp8_r1_{i:02d}' for i in range(10)] + [f'lbp16_r2_{i:02d}' for i in range(18)]
F4_NAMES = (['circularity', 'solidity', 'aspect_ratio_rotated', 'rotated_extent']
            + [f'radial_spectrum_{i:02d}' for i in range(1, 13)]
            + [f'iap_ring{r}_angle{a}' for r in range(1, 5) for a in range(4)]
            + [f'iep_ring{r}_density' for r in range(1, 6)] + ['iep_slope', 'iep_cv'])
FEATURE_NAMES = F1_NAMES + F2_NAMES + F3_NAMES + F4_NAMES
HISTOGRAM_SLICES = (slice(0, 72), slice(72, 78), slice(96, 106), slice(106, 124))


def validate_features(values):
    array = np.asarray(values)
    if array.ndim not in (1, 2) or array.shape[-1] != 163 or not np.isfinite(array).all():
        raise ValueError('Expected finite 163-dimensional feature vectors.')
    if array.ndim == 2 and not len(array):
        raise ValueError('The feature matrix is empty.')
    for part in HISTOGRAM_SLICES:
        hist = array[..., part]
        if np.any(hist < 0) or not np.allclose(hist.sum(axis=-1), 1, rtol=0, atol=2e-6):
            raise ValueError('Each HS, V, LBP8 and LBP16 histogram must sum to one.')
    return array


def feature_columns(vector):
    """Lossless conversion to the existing PostgreSQL feature columns."""
    v = validate_features(vector).tolist()
    return (v[:72], v[72:78], v[78:96:2], v[79:96:2], v[96:124],
            *v[124:128], v[128:140], v[140:156], v[156:163])


def vector_from_columns(columns):
    hs, value, means, stds, lbp, circularity, solidity, aspect, extent, radial, iap, iep = columns
    lab = np.column_stack((means, stds)).ravel().tolist()
    return np.asarray(hs + value + lab + lbp + [circularity, solidity, aspect, extent]
                      + radial + iap + iep, dtype=np.float32)
