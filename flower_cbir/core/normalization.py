import numpy as np


class SelectiveNormalizer:
    """
    Selective Normalization for 163-dimensional flower features:
    - F1 [0:78] (HSV Histograms) and F3 [96:124] (LBP Histograms) are kept unchanged (mean=0, std=1).
    - F2 [78:96] (Lab Ring Color stats) and F4 [124:163] (Shape & Structure) are Z-score normalized.
    """
    def __init__(self, mean: np.ndarray = None, std: np.ndarray = None, epsilon: float = 1e-12):
        self.epsilon = epsilon
        if mean is not None and std is not None:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
        else:
            self.mean = np.zeros(163, dtype=np.float32)
            self.std = np.ones(163, dtype=np.float32)

    def fit(self, raw_features_matrix: np.ndarray):
        """
        Fit mean and std with ddof=0 on valid dataset feature vectors.
        Shape of raw_features_matrix: (N, 163).
        """
        raw_features_matrix = np.asarray(raw_features_matrix)
        if raw_features_matrix.ndim != 2 or len(raw_features_matrix) == 0 or not np.isfinite(raw_features_matrix).all():
            raise ValueError('Normalization requires a nonempty finite feature matrix.')
        N, D = raw_features_matrix.shape
        if D != 163:
            raise ValueError(f"Expected 163 dimensions, got {D}")

        calc_mean = np.mean(raw_features_matrix, axis=0, dtype=np.float64)
        calc_std = np.std(raw_features_matrix, axis=0, ddof=0, dtype=np.float64)

        # Base identity: mean 0, std 1
        final_mean = np.zeros(163, dtype=np.float32)
        final_std = np.ones(163, dtype=np.float32)

        # F2: slice [78:96]
        final_mean[78:96] = calc_mean[78:96]
        f2_std = calc_std[78:96]
        final_std[78:96] = np.where(f2_std <= self.epsilon, 1.0, f2_std)

        # F4: slice [124:163]
        final_mean[124:163] = calc_mean[124:163]
        f4_std = calc_std[124:163]
        final_std[124:163] = np.where(f4_std <= self.epsilon, 1.0, f4_std)

        self.mean = final_mean
        self.std = final_std
        return self

    def transform(self, vector_or_matrix: np.ndarray) -> np.ndarray:
        """
        Apply selective normalization: z = (x - mean) / std.
        Works for 1D (163,) or 2D (N, 163).
        """
        values = np.asarray(vector_or_matrix)
        if values.ndim not in (1, 2) or values.shape[-1] != 163 or not np.isfinite(values).all():
            raise ValueError('Expected finite 163D vectors.')
        if self.mean.shape != (163,) or self.std.shape != (163,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid normalization parameters.')
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "dims": 163,
            "epsilon": self.epsilon
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]), epsilon=data.get("epsilon", 1e-12))
