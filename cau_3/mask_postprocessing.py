
from __future__ import annotations

import cv2
import numpy as np


def threshold_probabilities(prob, threshold=0.5):
    """prob: HxW, float, xác suất lớp hoa trong [0,1]."""
    prob = np.asarray(prob)

    if prob.ndim != 2:
        raise ValueError("Expected a 2D probability map")

    if not np.issubdtype(prob.dtype, np.floating):
        raise TypeError("Expected float probability map")

    if not np.all(np.isfinite(prob)):
        raise ValueError("Probability map contains NaN/Inf")

    if np.any(prob < 0) or np.any(prob > 1):
        raise ValueError("Input must contain probabilities [0,1]")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")

    return (prob >= threshold).astype(np.uint8) * 255


def morphology(mask, operation, size=3):
    if size <= 1:
        return mask.copy()

    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (size, size)
    )

    return cv2.morphologyEx(mask, operation, kernel)


def filter_components(
    mask,
    min_area_ratio=0.03,
    max_distance_ratio=0.20
):
    """
    Giữ thành phần chính và các vùng phụ đủ lớn,
    nằm gần thành phần chính.

    Các ngưỡng là heuristic cần kiểm chứng.
    """
    n, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    result = np.zeros_like(mask)

    if n <= 1:
        return result

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_id = 1 + int(np.argmax(areas))
    main_area = int(stats[main_id, cv2.CC_STAT_AREA])

    h, w = mask.shape
    diagonal = np.hypot(w, h)

    main_center = centers[main_id]
    kept = [main_id]

    for i in range(1, n):
        if i == main_id:
            continue

        area = stats[i, cv2.CC_STAT_AREA]

        if area < main_area * min_area_ratio:
            continue

        distance = np.linalg.norm(
            centers[i] - main_center
        ) / diagonal

        if distance <= max_distance_ratio:
            kept.append(i)

    result[np.isin(labels, kept)] = 255

    return result


def fill_small_holes(mask, max_area=100):
    """
    Chỉ lấp những vùng nền nhỏ được bao kín.
    Không lấp vùng nền thông ra mép ảnh.
    """
    background = (mask == 0).astype(np.uint8)

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            background, connectivity=8
        )
    )

    h, w = mask.shape
    result = mask.copy()

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        touches_border = (
            x == 0 or y == 0
            or x + width >= w
            or y + height >= h
        )

        if not touches_border and area <= max_area:
            result[labels == i] = 255

    return result


def repair_center_holes(
    mask,
    image_bgr,
    probability,
    center_radius_ratio=0.35,
    max_area_ratio=0.25,
    max_brightness=90,
    min_prob=0.10,
):
    """
    Phục hồi các lỗ rỗng kín ở vùng tâm hoa (nhụy tối màu bị nhận nhầm thành nền):
    - Lỗ khép kín bên trong mask (không thông ra mép ảnh).
    - Nằm gần tâm bông hoa (khoảng cách <= center_limit).
    - Có độ sáng thấp (nhụy tối màu).
    - Có xác suất hoa tối thiểu từ DeepLab.
    Không sửa vùng nền nối ra biên ảnh.
    """
    if mask.ndim != 2:
        raise ValueError("Mask phải là ảnh 2 chiều")
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError("Ảnh gốc và mask khác kích thước")
    if probability.shape != mask.shape:
        raise ValueError("Probability và mask khác kích thước")

    result = mask.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result

    # Bounding box của vùng hoa
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    flower_radius = min(
        x_max - x_min + 1,
        y_max - y_min + 1
    ) / 2
    center_limit = center_radius_ratio * 2 * flower_radius

    # Chuyển ảnh sang ảnh xám
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Phân tích những vùng nền nằm trong mask
    background = (mask == 0).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(
        background, connectivity=8
    )
    h, w = mask.shape
    flower_area = np.count_nonzero(mask)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        # Bỏ vùng quá lớn
        if area > flower_area * max_area_ratio:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Không lấp nền thông ra mép ảnh
        if (
            x == 0 or y == 0
            or x + bw >= w
            or y + bh >= h
        ):
            continue

        # Kiểm tra khoảng cách từ tâm lỗ đến tâm hoa
        hole_cx, hole_cy = centers[i]
        dist = np.hypot(hole_cx - cx, hole_cy - cy)
        if dist > center_limit:
            continue

        hole_mask = (labels == i)
        mean_brightness = float(gray[hole_mask].mean())
        mean_prob = float(probability[hole_mask].mean())

        # Điều kiện: nhụy tối màu (độ sáng thấp) và có xác suất hoa tối thiểu
        if mean_brightness <= max_brightness and mean_prob >= min_prob:
            result[hole_mask] = 255

    return result


def postprocess_mask(
    probability,
    image_bgr=None,
    threshold=0.45,
    opening=False,
    closing=False,
    fill_holes=False,
    keep_nearby=True,
    repair_center=False,
    max_hole_area=20,
):
    # 1. DeepLab probability -> binary mask
    raw = threshold_probabilities(
        probability, threshold
    )

    current = raw.copy()

    # 2. Opening nhẹ
    if opening:
        current = morphology(
            current, cv2.MORPH_OPEN, 3
        )

    # 3. Lọc thành phần liên thông
    if keep_nearby:
        current = filter_components(current)

    # 4. Lấp lỗ nhỏ
    if fill_holes:
        current = fill_small_holes(
            current, max_area=max_hole_area
        )

    # 4b. Phục hồi lỗ vùng nhụy tâm hoa
    if repair_center:
        if image_bgr is None:
            raise ValueError(
                "Cần image_bgr khi bật repair_center"
            )
        current = repair_center_holes(
            mask=current,
            image_bgr=image_bgr,
            probability=probability,
        )

    # 5. Closing tùy chọn
    if closing:
        current = morphology(
            current, cv2.MORPH_CLOSE, 3
        )

    # Mask hình dạng: luôn nhị phân
    binary_mask = current

    # Mask dùng hiển thị cutout: alpha mềm
    alpha_mask = cv2.GaussianBlur(
        binary_mask, (3, 3), 0
    )

    metrics = {
        "raw_pixels": int(np.count_nonzero(raw)),
        "final_pixels": int(
            np.count_nonzero(binary_mask)
        ),
        "pixel_delta": int(
            np.count_nonzero(binary_mask)
            - np.count_nonzero(raw)
        ),
        "changed_pixels": int(
            np.count_nonzero(raw != binary_mask)
        ),
        "threshold": threshold,
    }

    return binary_mask, alpha_mask, metrics
