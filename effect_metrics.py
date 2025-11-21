from metric_base import Metric

class IOIMetric(Metric):
    def compute(self, ld_patched: float, ld_clean: float, ld_corrupted: float) -> float:
        return (ld_patched - ld_corrupted) / ld_clean

class AttPMetric(Metric):
    def compute(self, ld_patched: float, ld_clean: float, ld_corrupted: float) -> float:
        return (ld_patched - ld_corrupted) / (ld_clean - ld_corrupted)