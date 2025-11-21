from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def compute(self, ld_patched: float, ld_clean: float, ld_corrupted: float) -> float:
        raise NotImplementedError
