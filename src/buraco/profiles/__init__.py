"""Rule profiles: named constructors returning configured RulesConfig trees."""

from __future__ import annotations

from typing import Callable

from buraco.config import RulesConfig
from buraco.profiles.biriba import biriba
from buraco.profiles.buraco import buraco
from buraco.profiles.canasta import canasta
from buraco.profiles.rummy import rummy

PROFILES: dict[str, Callable[..., RulesConfig]] = {
    "buraco": buraco,
    "canasta": canasta,
    "rummy": rummy,
    "biriba": biriba,
}


def load_profile(name: str, **kwargs) -> RulesConfig:
    try:
        return PROFILES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; available: {sorted(PROFILES)}") from None


__all__ = ["PROFILES", "biriba", "buraco", "canasta", "load_profile", "rummy"]
