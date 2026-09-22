"""Small, explicit experiment protocol; all knobs are captured in the run."""

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ResearchConfig:
    direction: str = "寻找跨资产中期动量与风险调整因子，控制换手和交易成本"
    mode: str = "local"
    trials: int = 24
    seed: int = 42
    cost_bps: float = 10.0
    top_k: int = 3
    rebalance_every: int = 5
    holdout_fraction: float = 0.2
    folds: int = 3
    warmup: int = 128
    max_seconds: int = 300
    max_llm_calls: int = 12
    max_llm_tokens: int = 24000
    max_complexity: int = 60
    dataset: str = "demo"
    constraints: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in {"local", "llm", "agent"}:
            raise ValueError("mode must be local, llm or agent")
        for name, low, high in [
            ("trials", 1, 200), ("seed", 0, 2**32 - 1), ("top_k", 1, 100),
            ("rebalance_every", 1, 63), ("folds", 2, 6), ("warmup", 64, 252),
            ("max_seconds", 1, 3600), ("max_llm_calls", 0, 100),
            ("max_llm_tokens", 0, 2000000 if self.mode == "agent" else 500000), ("max_complexity", 3, 150),
        ]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, low, high in [("cost_bps", 0, 200), ("holdout_fraction", 0.1, 0.4)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite in [{low}, {high}]")
        if not isinstance(self.direction, str) or not 1 <= len(self.direction.strip()) <= 2000:
            raise ValueError("direction must be 1–2000 characters")
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a string")
        if not isinstance(self.constraints, dict) or set(self.constraints) - {"required_window", "forbidden_fields", "max_turnover"}:
            raise ValueError("Unknown research constraints")
        if self.constraints and self.mode != "agent":
            raise ValueError("Executable research constraints require agent mode")
        window = self.constraints.get("required_window")
        if window is not None and (type(window) is not int or not 3 <= window <= 60):
            raise ValueError("required_window must be an integer from 3 to 60")
        fields = self.constraints.get("forbidden_fields", [])
        if not isinstance(fields, list) or any(value not in {"open", "high", "low", "close", "volume"} for value in fields):
            raise ValueError("forbidden_fields must name OHLCV fields")
        turnover = self.constraints.get("max_turnover")
        if turnover is not None and (type(turnover) not in {float, int} or not math.isfinite(turnover) or not 0 < turnover <= 2):
            raise ValueError("max_turnover must be a fraction in (0,2]")

    def to_dict(self):
        return asdict(self)
