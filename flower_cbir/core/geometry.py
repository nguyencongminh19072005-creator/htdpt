import cv2
import numpy as np
from flower_cbir.core.segmentation import clean_mask
from flower_cbir.settings import load_config


def compute_geometry(mask_binary: np.ndarray, epsilon: float = 1e-12) -> dict:
    """
    Compute geometric properties and concentric ring maps from binary mask M (384x384).
    The mask uses the shared largest-component policy.
    - A: number of pixels where M == 1.
    - Centroid (xc, yc): center of mass of M.
    - M_int: eroded M with 5x5 ellipse kernel (B5).
    - r_max: max Euclidean distance from (xc, yc) to any pixel in M_int.
    - rho: distance / r_max.
    - F2 ring map (3 rings on M): min(floor(3 * rho), 2).
    - IAP ring map (4 rings on M_int): min(floor(4 * rho), 3).
    - IEP ring map (5 rings on M_int): min(floor(5 * rho), 4).
    """
    mask_binary = clean_mask(mask_binary)
    h, w = mask_binary.shape[:2]
    m_bool = mask_binary > 127
    A = float(np.sum(m_bool))

    if A == 0:
        raise ValueError("Degenerate mask: Foreground area A = 0.")

    # Image moments for exact center of mass
    moments = cv2.moments(mask_binary)
    if moments["m00"] > 0:
        xc = float(moments["m10"] / moments["m00"])
        yc = float(moments["m01"] / moments["m00"])
    else:
        # Fallback to mean coordinates
        y_idxs, x_idxs = np.where(m_bool)
        xc = float(np.mean(x_idxs))
        yc = float(np.mean(y_idxs))

    # Internal mask M_int = erode(M, B5)
    geometry_config = load_config()['geometry']
    kernel_b5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(geometry_config['b5_kernel_size']))
    m_int_binary = cv2.erode(mask_binary, kernel_b5, iterations=geometry_config['erode_iterations'], borderType=cv2.BORDER_CONSTANT, borderValue=0)
    m_int_bool = m_int_binary > 127
    A_int = float(np.sum(m_int_bool))
    if A_int == 0:
        raise ValueError('Mask is too thin: erosion removes all foreground.')

    # Grid of coordinates
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_map = np.hypot(x_grid - xc, y_grid - yc)

    # r_max computed on M_int
    if A_int > 0:
        r_max = float(np.max(dist_map[m_int_bool]))
    else:
        # Fallback if erosion removed entire mask (e.g. very thin petals)
        r_max = float(np.max(dist_map[m_bool]))

    if r_max <= epsilon:
        r_max = 1.0

    rho_map = dist_map / r_max

    # Ring mappings:
    # 3 rings for F2 on M: 0, 1, 2
    f2_ring_map = np.clip(np.floor(3.0 * rho_map).astype(np.int32), 0, 2)

    # 4 rings for IAP on M_int: 0, 1, 2, 3
    iap_ring_map = np.clip(np.floor(4.0 * rho_map).astype(np.int32), 0, 3)

    # 5 rings for IEP on M_int: 0, 1, 2, 3, 4
    iep_ring_map = np.clip(np.floor(5.0 * rho_map).astype(np.int32), 0, 4)

    # Contours and bounding shapes
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No external contour found in mask.")
    main_contour = max(contours, key=cv2.contourArea)

    perimeter = float(cv2.arcLength(main_contour, True))
    hull = cv2.convexHull(main_contour)
    hull_area = float(cv2.contourArea(hull))

    min_rect = cv2.minAreaRect(main_contour)
    (rect_cx, rect_cy), (rw, rh), rect_angle = min_rect
    side_max = max(rw, rh)
    side_min = min(rw, rh)

    return {
        "A": A,
        "A_int": A_int,
        "centroid": (xc, yc),
        "r_max": r_max,
        "m_bool": m_bool,
        "m_int_bool": m_int_bool,
        "m_int_binary": m_int_binary,
        "dist_map": dist_map,
        "rho_map": rho_map,
        "f2_ring_map": f2_ring_map,
        "iap_ring_map": iap_ring_map,
        "iep_ring_map": iep_ring_map,
        "main_contour": main_contour,
        "perimeter": perimeter,
        "hull": hull,
        "hull_area": hull_area,
        "min_rect": min_rect,
        "side_max": side_max,
        "side_min": side_min,
    }
