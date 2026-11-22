import numpy as np
from flower_cbir.settings import validate_weights


def hellinger_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Hellinger distance between two probability distributions p and q.
    H(p, q) = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))
    Supports 1D vs 1D or 1D vs 2D (matrix of distributions).
    """
    p_safe = np.maximum(p, 0.0)
    q_safe = np.maximum(q, 0.0)
    diff = np.sqrt(p_safe) - np.sqrt(q_safe)
    # If q is 2D (N, D), sum over axis 1
    axis = -1 if diff.ndim > 1 else 0
    return np.sqrt(0.5 * np.sum(diff ** 2, axis=axis))


def rms_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    RMS distance transformed to [0, 1):
    R(a, b) = sqrt(mean((a - b)^2))
    T(a, b) = R(a, b) / (1.0 + R(a, b))
    Supports 1D vs 1D or 1D vs 2D (matrix).
    """
    diff = a - b
    axis = -1 if diff.ndim > 1 else 0
    rms = np.sqrt(np.mean(diff ** 2, axis=axis))
    return rms / (1.0 + rms)


def compute_group_distances(query_norm: np.ndarray, target_norm: np.ndarray):
    """
    Compute D1, D2, D3, D4 components between query_norm and target_norm.
    query_norm: shape (163,)
    target_norm: shape (163,) or (N, 163)
    Returns:
        D1, D2, D3, D4, and details dict.
    """
    # F1 components: HS [0:72], V [72:78]
    hs_q, hs_t = query_norm[0:72], target_norm[..., 0:72]
    v_q, v_t = query_norm[72:78], target_norm[..., 72:78]
    d_hs = hellinger_dist(hs_q, hs_t)
    d_v = hellinger_dist(v_q, v_t)
    D1 = 0.5 * (d_hs + d_v)

    # F2 component: Lab 3 rings [78:96]
    f2_q, f2_t = query_norm[78:96], target_norm[..., 78:96]
    D2 = rms_transform(f2_q, f2_t)

    # F3 components: LBP8 [96:106], LBP16 [106:124]
    lbp8_q, lbp8_t = query_norm[96:106], target_norm[..., 96:106]
    lbp16_q, lbp16_t = query_norm[106:124], target_norm[..., 106:124]
    d_lbp8 = hellinger_dist(lbp8_q, lbp8_t)
    d_lbp16 = hellinger_dist(lbp16_q, lbp16_t)
    D3 = 0.5 * (d_lbp8 + d_lbp16)

    # F4 components (average of 4 blocks):
    # - scalars: [124:128]
    # - fourier: [128:140]
    # - iap: [140:156]
    # - iep: [156:163]
    d4_scalars = rms_transform(query_norm[124:128], target_norm[..., 124:128])
    d4_fourier = rms_transform(query_norm[128:140], target_norm[..., 128:140])
    d4_iap = rms_transform(query_norm[140:156], target_norm[..., 140:156])
    d4_iep = rms_transform(query_norm[156:163], target_norm[..., 156:163])
    D4 = 0.25 * (d4_scalars + d4_fourier + d4_iap + d4_iep)

    details = {
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "D4": D4,
        "d_hs": d_hs,
        "d_v": d_v,
        "d_lbp8": d_lbp8,
        "d_lbp16": d_lbp16,
        "d4_scalars": d4_scalars,
        "d4_fourier": d4_fourier,
        "d4_iap": d4_iap,
        "d4_iep": d4_iep,
    }

    return D1, D2, D3, D4, details


def compute_overall_distance(query_norm: np.ndarray, target_norm: np.ndarray,
                             weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculate combined distance D and similarity S = 1 - D.
    D = w1*D1 + w2*D2 + w3*D3 + w4*D4
    """
    query_norm = np.asarray(query_norm)
    target_norm = np.asarray(target_norm)
    if query_norm.shape != (163,) or target_norm.ndim not in (1, 2) or target_norm.shape[-1] != 163:
        raise ValueError('Distance requires a 163D query and 163D targets.')
    if not np.isfinite(query_norm).all() or not np.isfinite(target_norm).all():
        raise ValueError('Distance inputs must be finite.')
    w1, w2, w3, w4 = validate_weights(weights)
    D1, D2, D3, D4, details = compute_group_distances(query_norm, target_norm)
    D = w1 * D1 + w2 * D2 + w3 * D3 + w4 * D4
    S = np.clip(1.0 - D, 0.0, 1.0)
    return D, S, details
