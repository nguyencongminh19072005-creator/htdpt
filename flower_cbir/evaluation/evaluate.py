import time
import numpy as np

from flower_cbir.core.distance import compute_overall_distance
from flower_cbir.services.search import FlowerSearchEngine
from flower_cbir.settings import load_config


def run_correctness_checks(engine: FlowerSearchEngine):
    """
    Check mathematical invariants and vector properties across the dataset index:
    1. Shape: N x 163
    2. Sum of HS == 1.0 (within 1e-4)
    3. Sum of V == 1.0
    4. Sum of LBP8 == 1.0
    5. Sum of LBP16 == 1.0
    6. Self-distance on sample images ~ 0.0
    """
    raw = engine.raw_matrix
    norm = engine.normalized_matrix
    N, D = raw.shape

    report = {"total_images": N, "total_dims": D, "tests": []}

    # Test 1: Dimensions
    dim_ok = (D == 163 and N == load_config()['dataset']['expected_count'] and norm.shape == raw.shape and len(engine.metadata) == N)
    report["tests"].append({
        "name": "Dimension Check (163D)",
        "passed": dim_ok,
        "detail": f"Expected {load_config()['dataset']['expected_count']} x 163, got {N} x {D}"
    })

    # Test 2: F1 and F3 histogram sums
    hs_sums = np.sum(raw[:, 0:72], axis=1)
    v_sums = np.sum(raw[:, 72:78], axis=1)
    lbp8_sums = np.sum(raw[:, 96:106], axis=1)
    lbp16_sums = np.sum(raw[:, 106:124], axis=1)

    hs_ok = bool(np.all(np.abs(hs_sums - 1.0) < 1e-4))
    v_ok = bool(np.all(np.abs(v_sums - 1.0) < 1e-4))
    lbp8_ok = bool(np.all(np.abs(lbp8_sums - 1.0) < 1e-4))
    lbp16_ok = bool(np.all(np.abs(lbp16_sums - 1.0) < 1e-4))

    report["tests"].append({
        "name": "Histogram Sum Invariant (HS=1, V=1, LBP8=1, LBP16=1)",
        "passed": hs_ok and v_ok and lbp8_ok and lbp16_ok,
        "detail": f"HS: {np.mean(hs_sums):.4f}, V: {np.mean(v_sums):.4f}, LBP8: {np.mean(lbp8_sums):.4f}, LBP16: {np.mean(lbp16_sums):.4f}"
    })

    # Test 3: Normalization unchanged for F1 & F3
    f1_unchanged = bool(np.allclose(raw[:, 0:78], norm[:, 0:78], atol=1e-5))
    f3_unchanged = bool(np.allclose(raw[:, 96:124], norm[:, 96:124], atol=1e-5))
    report["tests"].append({
        "name": "Selective Normalization Invariant (F1 & F3 unchanged)",
        "passed": f1_unchanged and f3_unchanged,
        "detail": f"F1 unchanged: {f1_unchanged}, F3 unchanged: {f3_unchanged}"
    })

    # Test 4: Self-distance on first 10 images
    self_distances = []
    for i in range(N):
        vec = norm[i]
        D_self, S_self, _ = compute_overall_distance(vec, vec)
        self_distances.append(float(D_self))

    max_self_d = max(self_distances) if self_distances else 0.0
    self_ok = max_self_d < 1e-4
    report["tests"].append({
        "name": "Self-Distance Verification (D(q, q) ~ 0.0)",
        "passed": self_ok,
        "detail": f"Max self distance across {N} samples: {max_self_d:.6f}"
    })
    report['tests'].append({'name': 'Finite vectors and metadata uniqueness', 'passed': bool(np.isfinite(raw).all() and np.isfinite(norm).all() and len(set(engine.image_ids.tolist())) == N), 'detail': f'{N} finite vectors and unique image IDs'})
    reproduced = engine.normalizer.transform(raw)
    report['tests'].append({'name': 'Normalizer reproduces index', 'passed': bool(np.array_equal(reproduced, norm)), 'detail': f'Max error: {float(np.max(np.abs(reproduced - norm))):.8f}'})

    return report


def run_ablation_study(engine: FlowerSearchEngine, query_idx: int = 0):
    """
    Perform ablation study on different feature groups:
    - Baseline (all 4 groups, equal weights 0.25)
    - Only F1 (Color)
    - Only F2 (Regional Lab)
    - Only F3 (Texture LBP)
    - Only F4 (Shape & Internal Structure)
    - F1 + F2 (Color overall + Regional)
    - Drop F1
    - Drop F2
    - Drop F3
    - Drop F4
    """
    query_vec = engine.normalized_matrix[query_idx]
    query_filename = engine.metadata[query_idx]["filename"]

    configs = {
        "Full 163D Baseline": (0.25, 0.25, 0.25, 0.25),
        "Only F1 (Color)": (1.0, 0.0, 0.0, 0.0),
        "Only F2 (Regional Lab)": (0.0, 1.0, 0.0, 0.0),
        "Only F3 (Texture LBP)": (0.0, 0.0, 1.0, 0.0),
        "Only F4 (Shape & Structure)": (0.0, 0.0, 0.0, 1.0),
        "Color Only (F1 + F2)": (0.5, 0.5, 0.0, 0.0),
        "Structure & Texture (F3 + F4)": (0.0, 0.0, 0.5, 0.5),
        "Ablate F1 (No Overall Color)": (0.0, 1/3, 1/3, 1/3),
        "Ablate F2 (No Regional Color)": (1/3, 0.0, 1/3, 1/3),
        "Ablate F3 (No Texture)": (1/3, 1/3, 0.0, 1/3),
        "Ablate F4 (No Shape)": (1/3, 1/3, 1/3, 0.0),
    }

    results = {}
    for name, w in configs.items():
        res, t_ms = engine.search_by_vector(
            query_vec, weights=w, top_k=5, exclude_filename=query_filename
        )
        results[name] = {
            "weights": list(w),
            "search_ms": t_ms,
            "top5": [
                {"rank": r["rank"], "filename": r["filename"], "distance": round(r["distance"], 4), "similarity": round(r["similarity"], 4)}
                for r in res
            ]
        }

    return results
