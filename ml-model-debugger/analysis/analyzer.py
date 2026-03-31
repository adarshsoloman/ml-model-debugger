class ModelDiagnosis:
    UNDERFITTING = "Underfitting"
    OVERFITTING = "Overfitting"
    ACCEPTABLE = "Acceptable"
    UNKNOWN = "Unknown"

from core.config import Config

class Analyzer:
    def __init__(self, performance_threshold: float = None, overfitting_ratio: float = None):
        """
        Initialize the analyzer with heuristics.
        """
        self.performance_threshold = performance_threshold or Config.DEFAULT_PERFORMANCE_THRESHOLD
        self.overfitting_ratio = overfitting_ratio or Config.DEFAULT_OVERFITTING_RATIO

    def diagnose(self, train_metrics: dict, val_metrics: dict) -> str:
        """
        Compare training and validation metrics to diagnose model behavior.
        We primarily look at RMSE and R2.
        """
        train_rmse = train_metrics.get("rmse", 0)
        val_rmse = val_metrics.get("rmse", 0)
        train_r2 = train_metrics.get("r2", 0)

        # 1. Check for Underfitting: Model performs poorly even on training data
        if train_r2 < self.performance_threshold:
            return ModelDiagnosis.UNDERFITTING

        # 2. Check for Overfitting: High variance between train and validation error
        if train_rmse > 0 and (val_rmse / train_rmse) > self.overfitting_ratio:
            return ModelDiagnosis.OVERFITTING

        return ModelDiagnosis.ACCEPTABLE
