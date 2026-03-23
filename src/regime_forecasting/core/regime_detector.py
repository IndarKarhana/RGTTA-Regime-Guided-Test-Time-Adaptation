import numpy as np
from scipy import stats


class RegimeDetector:
    """Simple regime detector compatible with tests.

    Stores `feature_history` (list of feature vectors) and `regime_flags` (list of bools).
    """

    def __init__(self, season_length: int = 12, regime_threshold: float = 0.7):
        self.season_length = season_length
        self.regime_threshold = regime_threshold
        self.feature_history = []
        self.regime_flags = []

    def extract_regime_features(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        # Basic features: mean, std, skew, kurtosis, autocorr lag1
        mean = float(np.mean(y))
        std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
        skew = float(stats.skew(y)) if len(y) > 2 else 0.0
        kurt = float(stats.kurtosis(y)) if len(y) > 3 else 0.0
        if len(y) > 1:
            acf1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        else:
            acf1 = 0.0
        features = np.array([mean, std, skew, kurt, acf1], dtype=float)
        # Ensure finite
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def detect_regime_change(self, y: np.ndarray, use_flag_method: bool = False):
        features = self.extract_regime_features(y)
        is_change = False
        # If no history, first window is considered a change
        if len(self.feature_history) == 0:
            is_change = True
        else:
            prev = self.feature_history[-1]
            # If flag method requested, check mean drop
            if use_flag_method:
                prev_mean = prev[0] if len(prev) > 0 else 0.0
                cur_mean = features[0]
                if prev_mean > 0 and cur_mean < self.regime_threshold * prev_mean:
                    is_change = True
                else:
                    is_change = False
            else:
                # Cosine similarity as crude measure
                num = np.dot(features, prev)
                denom = (np.linalg.norm(features) * np.linalg.norm(prev)) + 1e-8
                sim = float(num / denom)
                # Consider change if similarity low
                is_change = sim < 0.8

        # Record
        self.feature_history.append(features)
        self.regime_flags.append(bool(is_change))
        return bool(is_change), features

    def get_detection_summary(self):
        total = len(self.feature_history)
        changes = int(sum(self.regime_flags))
        change_rate = float(changes / total) if total > 0 else 0.0
        return {"total_windows": total, "regime_changes": changes, "change_rate": change_rate}
