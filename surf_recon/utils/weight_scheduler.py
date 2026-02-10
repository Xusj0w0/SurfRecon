from dataclasses import dataclass
from typing import Optional


@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: Optional[int] = None
