from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    intercept_gain: float = 1.52
    velocity_gain: float = 2.0
    search_lateral_gain: float = 0.76
    search_vertical_gain: float = 0.20
    search_forward_gain: float = 0.56
    lead_gain: float = 10.0
    damping_gain: float = 0.15
    max_action: float = 1.0


POLICY_CONFIG = PolicyConfig()
