"""Pure-Python helpers for deterministic evaluation-suite configuration.

This module deliberately has no Isaac Lab imports so suite validation and rubric
logic can be tested on machines that do not have Isaac Sim installed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import yaml

REQUIRED_CONDITIONS = (
    "initial_xy_offset_range_m",
    "initial_yaw_error_range_deg",
    "peg_friction_range",
    "socket_friction_range",
    "peg_mass_scale_range",
)


def load_named_yaml(path: str | Path, name: str, *, kind: str) -> dict:
    """Load and validate one named mapping from a YAML file."""
    with open(path) as stream:
        entries = yaml.safe_load(stream)
    if not isinstance(entries, Mapping) or name not in entries:
        available = list(entries) if isinstance(entries, Mapping) else []
        raise ValueError(f"Unknown {kind} {name!r}. Available: {available}")
    entry = entries[name]
    if not isinstance(entry, Mapping):
        raise ValueError(f"{kind.title()} {name!r} must be a mapping")
    return dict(entry)


def validate_suite(name: str, suite: Mapping) -> dict:
    """Return a normalized suite or raise with an actionable configuration error."""
    episodes = suite.get("episodes")
    seeds = suite.get("seeds")
    conditions = suite.get("conditions")
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError(f"Suite {name!r}: episodes must be a positive integer")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"Suite {name!r}: seeds must be a non-empty list of integers")
    if not isinstance(conditions, Mapping):
        raise ValueError(f"Suite {name!r}: conditions must be a mapping")

    normalized_conditions = {}
    for key in REQUIRED_CONDITIONS:
        if key not in conditions:
            raise ValueError(f"Suite {name!r}: missing conditions.{key}")
        normalized_conditions[key] = _validate_range(name, key, conditions[key])

    for key in ("initial_xy_offset_range_m", "initial_yaw_error_range_deg"):
        low, high = normalized_conditions[key]
        if not math.isclose(low, -high, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Suite {name!r}: {key} must be symmetric about zero because Factory/Forge "
                "samples pose noise as +/- an amplitude"
            )
    for key in ("peg_friction_range", "socket_friction_range", "peg_mass_scale_range"):
        if normalized_conditions[key][0] <= 0:
            raise ValueError(f"Suite {name!r}: {key} values must be positive")

    return {"episodes": episodes, "seeds": list(seeds), "conditions": normalized_conditions}


def apply_suite_to_env_cfg(env_cfg, suite: Mapping) -> None:
    """Apply validated pose and dynamics conditions before ``gym.make``.

    Pose noise is handled by Factory's reset routine. Mass uses Isaac Lab's
    reset event. Friction ranges are stored on ForgeEnvCfg and are applied by
    ForgeEnv *after* FactoryEnv overwrites startup material randomization.
    """
    conditions = suite["conditions"]
    xy_amplitude = max(abs(value) for value in conditions["initial_xy_offset_range_m"])
    yaw_amplitude_deg = max(abs(value) for value in conditions["initial_yaw_error_range_deg"])

    env_cfg.task.hand_init_pos_noise[0:2] = [xy_amplitude, xy_amplitude]
    env_cfg.task.hand_init_orn_noise[2] = math.radians(yaw_amplitude_deg)
    # Make the requested XY range authoritative rather than adding the default
    # asset-in-gripper translation noise on top of it.
    env_cfg.task.held_asset_pos_noise[0:2] = [0.0, 0.0]

    env_cfg.events.object_scale_mass.params["mass_distribution_params"] = tuple(conditions["peg_mass_scale_range"])
    env_cfg.events.object_scale_mass.params["operation"] = "scale"
    env_cfg.events.object_scale_mass.params["distribution"] = "uniform"

    env_cfg.evaluation_peg_friction_range = tuple(conditions["peg_friction_range"])
    env_cfg.evaluation_socket_friction_range = tuple(conditions["socket_friction_range"])


def _validate_range(suite_name: str, key: str, value) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Suite {suite_name!r}: conditions.{key} must be a two-number list")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"Suite {suite_name!r}: conditions.{key} must be finite and ordered [low, high]")
    return low, high
