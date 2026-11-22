import cv2
import numpy as np
from functools import lru_cache
from skimage.feature import local_binary_pattern
from flower_cbir.core.geometry import compute_geometry
from flower_cbir.core.schema import validate_features


def extract_f1(image_rgb: np.ndarray, geom: dict, epsilon: float = 1e-12):
    """
    F1: Overall Color (78 dimensions)
    - HS 2D Histogram: 12 Hue bins [0, 360) x 6 Saturation bins [0, 1] = 72 dims
    - V Brightness Histogram: 6 bins [0, 1] = 6 dims
    Both normalized separately to sum to 1.
    """
    m_bool = geom["m_bool"]
    rgb_f32 = image_rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2HSV)

    H = hsv[:, :, 0][m_bool]  # [0, 360)
    S = hsv[:, :, 1][m_bool]  # [0, 1]
    V = hsv[:, :, 2][m_bool]  # [0, 1]

    num_pixels = len(H)
    if num_pixels == 0:
        return np.zeros(78, dtype=np.float32), {"hs_2d": np.zeros((12, 6)), "v_hist": np.zeros(6)}

    # HS 2D Histogram: 12 H bins x 6 S bins
    H_clipped = np.mod(H, 360.0)
    h_bins = np.linspace(0.0, 360.0, 13)
    s_bins = np.linspace(0.0, 1.0, 7)

    hs_2d, _, _ = np.histogram2d(H_clipped, S, bins=[h_bins, s_bins])
    hs_sum = float(np.sum(hs_2d))
    if hs_sum > 0:
        hs_norm = hs_2d / hs_sum
    else:
        hs_norm = hs_2d

    # Flatten according to h_index * 6 + s_index (row-major)
    hs_flat = hs_norm.flatten().astype(np.float32)

    # V 1D Histogram: 6 bins
    v_bins = np.linspace(0.0, 1.0, 7)
    v_hist, _ = np.histogram(V, bins=v_bins)
    v_sum = float(np.sum(v_hist))
    if v_sum > 0:
        v_norm = v_hist / v_sum
    else:
        v_norm = v_hist
    v_flat = v_norm.astype(np.float32)

    f1 = np.concatenate([hs_flat, v_flat]).astype(np.float32)
    return f1, {"hs_2d": hs_norm, "v_hist": v_flat}


def extract_f2(image_rgb: np.ndarray, geom: dict, epsilon: float = 1e-12):
    """
    F2: Color by Radius Ring in Lab space (18 dimensions)
    3 concentric rings on M (inner, middle, outer):
    Each ring: [mu_L, sigma_L, mu_a, sigma_a, mu_b, sigma_b] (ddof=0)
    """
    m_bool = geom["m_bool"]
    f2_ring_map = geom["f2_ring_map"]

    rgb_f32 = image_rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2Lab)

    f2_list = []
    ring_stats = []

    for k in range(3):
        ring_mask = m_bool & (f2_ring_map == k)
        count = int(np.sum(ring_mask))
        if count > 0:
            L_vals = lab[:, :, 0][ring_mask]
            a_vals = lab[:, :, 1][ring_mask]
            b_vals = lab[:, :, 2][ring_mask]

            mu_L, std_L = float(np.mean(L_vals)), float(np.std(L_vals, ddof=0))
            mu_a, std_a = float(np.mean(a_vals)), float(np.std(a_vals, ddof=0))
            mu_b, std_b = float(np.mean(b_vals)), float(np.std(b_vals, ddof=0))
        else:
            mu_L, std_L = 0.0, 0.0
            mu_a, std_a = 0.0, 0.0
            mu_b, std_b = 0.0, 0.0

        ring_vec = [mu_L, std_L, mu_a, std_a, mu_b, std_b]
        f2_list.extend(ring_vec)
        ring_stats.append({
            "ring": k,
            "count": count,
            "mu_L": mu_L, "std_L": std_L,
            "mu_a": mu_a, "std_a": std_a,
            "mu_b": mu_b, "std_b": std_b,
        })

    f2 = np.array(f2_list, dtype=np.float32)
    return f2, {"ring_stats": ring_stats}


def extract_f3(image_rgb: np.ndarray, geom: dict, epsilon: float = 1e-12, gray=None):
    """
    F3: Surface Texture (28 dimensions)
    Multi-radius rotation-invariant uniform LBP:
    - LBP(P=8, R=1, method='uniform') -> 10 bins [0..9]
    - LBP(P=16, R=2, method='uniform') -> 18 bins [0..17]
    Computed on gray image, aggregated only on flower pixels M.
    Each histogram normalized separately to sum to 1.
    """
    m_bool = geom["m_bool"]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if gray is None else gray

    # LBP 8, 1
    lbp8 = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp8_vals = lbp8[m_bool]
    hist8, _ = np.histogram(lbp8_vals, bins=np.arange(11))
    s8 = float(np.sum(hist8))
    norm8 = (hist8 / s8 if s8 > 0 else hist8).astype(np.float32)

    # LBP 16, 2
    lbp16 = local_binary_pattern(gray, P=16, R=2, method='uniform')
    lbp16_vals = lbp16[m_bool]
    hist16, _ = np.histogram(lbp16_vals, bins=np.arange(19))
    s16 = float(np.sum(hist16))
    norm16 = (hist16 / s16 if s16 > 0 else hist16).astype(np.float32)

    f3 = np.concatenate([norm8, norm16]).astype(np.float32)
    return f3, {"lbp8_hist": norm8, "lbp16_hist": norm16}


@lru_cache(maxsize=4)
def radial_sampling_grid(height, width):
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    steps = np.arange(0.0, np.hypot(width, height), 0.25)
    return steps, np.cos(angles)[:, None] * steps, np.sin(angles)[:, None] * steps


def extract_f4(image_rgb: np.ndarray, geom: dict, epsilon: float = 1e-12, gray=None):
    """
    F4: Shape and Internal Structure (39 dimensions)
    - 4 Scalars: Circularity, Solidity, Aspect Ratio, Rotated Extent
    - 12 Fourier Radial Spectrum components: S[1]..S[12]
    - 16 Internal Angular Pattern (IAP): 4 rings x 4 orientation bins
    - 7 Internal Edge Profile (IEP): d1..d5, Sedge, CVedge
    """
    A = geom["A"]
    P = geom["perimeter"]
    A_hull = geom["hull_area"]
    side_max = geom["side_max"]
    side_min = geom["side_min"]
    xc, yc = geom["centroid"]
    m_bool = geom["m_bool"]
    m_int_bool = geom["m_int_bool"]
    dist_map = geom["dist_map"]
    h, w = image_rgb.shape[:2]

    # --- 1. 4 Shape Scalars (4 dims) ---
    circularity = float(np.clip(4.0 * np.pi * A / (P * P + epsilon), 0.0, 1.0))
    solidity = float(np.clip(A / (A_hull + epsilon), 0.0, 1.0))
    aspect_ratio = float(side_max / (side_min + epsilon))
    rotated_extent = float(np.clip(A / (side_max * side_min + epsilon), 0.0, 1.0))
    scalars = np.array([circularity, solidity, aspect_ratio, rotated_extent], dtype=np.float32)

    # --- 2. Radial Spectrum (12 dims) ---
    # Sample 128 rays from centroid
    steps, offsets_x, offsets_y = radial_sampling_grid(h, w)
    sample_x = np.round(xc + offsets_x).astype(np.int32)
    sample_y = np.round(yc + offsets_y).astype(np.int32)
    valid = (sample_x >= 0) & (sample_x < w) & (sample_y >= 0) & (sample_y < h)
    inside = np.zeros(valid.shape, dtype=bool)
    inside[valid] = m_bool[sample_y[valid], sample_x[valid]]
    r_signal = np.max(np.where(inside, steps[None, :], 0.0), axis=1)

    fft_vals = np.fft.fft(r_signal)
    dc_mag = np.abs(fft_vals[0])
    if dc_mag > epsilon:
        radial_spectrum = (np.abs(fft_vals[1:13]) / dc_mag).astype(np.float32)
    else:
        radial_spectrum = np.zeros(12, dtype=np.float32)

    # --- 3. Internal Angular Pattern (IAP - 16 dims) ---
    # Sobel on original grayscale image (without mask to avoid artificial edge at mask border)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if gray is None else gray
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)

    G = np.hypot(gx, gy)
    theta_grad = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)

    y_grid, x_grid = np.indices((h, w))
    dx = x_grid - xc
    dy = y_grid - yc
    phi_rad = np.mod(np.degrees(np.arctan2(dy, dx)), 180.0)
    # At centroid dx=dy=0, convention phi = 0
    phi_rad[dist_map < epsilon] = 0.0

    psi = np.mod(theta_grad - phi_rad, 180.0)
    angle_bin = np.floor(np.mod(psi + 22.5, 180.0) / 45.0).astype(np.int32)
    angle_bin = np.clip(angle_bin, 0, 3)

    iap_ring_map = geom["iap_ring_map"]
    iap_matrix = np.zeros((4, 4), dtype=np.float32)

    for r in range(4):
        ring_mask = m_int_bool & (iap_ring_map == r)
        if np.any(ring_mask):
            G_ring = G[ring_mask]
            bin_ring = angle_bin[ring_mask]
            for m in range(4):
                iap_matrix[r, m] = float(np.sum(G_ring[bin_ring == m]))
            ring_sum = float(np.sum(iap_matrix[r, :]))
            iap_matrix[r, :] /= (ring_sum + epsilon)

    iap_flat = iap_matrix.flatten().astype(np.float32)

    # --- 4. Internal Edge Profile (IEP - 7 dims) ---
    # Smooth slightly before Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    canny = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)
    internal_edges = (canny > 0) & m_int_bool

    iep_ring_map = geom["iep_ring_map"]
    d_vals = np.zeros(5, dtype=np.float32)

    for k in range(5):
        ring_k = m_int_bool & (iep_ring_map == k)
        ring_area = float(np.sum(ring_k))
        if ring_area > 0:
            edge_count = float(np.sum(internal_edges & ring_k))
            d_vals[k] = edge_count / (ring_area + epsilon)
        else:
            d_vals[k] = 0.0

    # Sedge: center-weighted slope from inner (k=1) to outer (k=5)
    # sum((k - 3) * dk for k=1..5) / 10
    k_weights = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    s_edge = float(np.sum(k_weights * d_vals) / 10.0)

    # CVedge: std / mean of edge densities
    mean_d = float(np.mean(d_vals))
    std_d = float(np.std(d_vals, ddof=0))
    cv_edge = float(std_d / (mean_d + epsilon))

    iep = np.concatenate([d_vals, np.array([s_edge, cv_edge], dtype=np.float32)]).astype(np.float32)

    f4 = np.concatenate([scalars, radial_spectrum, iap_flat, iep]).astype(np.float32)

    explain_f4 = {
        "scalars": {
            "circularity": circularity,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "rotated_extent": rotated_extent,
        },
        "r_signal": r_signal,
        "radial_spectrum": radial_spectrum,
        "iap_matrix": iap_matrix,
        "d_vals": d_vals,
        "s_edge": s_edge,
        "cv_edge": cv_edge,
    }

    return f4, explain_f4


def extract_all_features(image_rgb: np.ndarray, mask_binary: np.ndarray, epsilon: float = 1e-12):
    """
    Extract the complete 163-dimensional feature vector and explanation data.
    """
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError('Features require a uint8 RGB image.')
    if image_rgb.shape[:2] != mask_binary.shape:
        raise ValueError('Image and mask must share the same coordinate system and shape.')
    geom = compute_geometry(mask_binary, epsilon=epsilon)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    f1, exp_f1 = extract_f1(image_rgb, geom, epsilon=epsilon)
    f2, exp_f2 = extract_f2(image_rgb, geom, epsilon=epsilon)
    f3, exp_f3 = extract_f3(image_rgb, geom, epsilon=epsilon, gray=gray)
    f4, exp_f4 = extract_f4(image_rgb, geom, epsilon=epsilon, gray=gray)

    vector_163 = np.concatenate([f1, f2, f3, f4]).astype(np.float32)
    validate_features(vector_163)

    explanation = {
        "geometry": geom,
        "f1": exp_f1,
        "f2": exp_f2,
        "f3": exp_f3,
        "f4": exp_f4,
        "raw_vector": vector_163,
    }

    return vector_163, explanation
