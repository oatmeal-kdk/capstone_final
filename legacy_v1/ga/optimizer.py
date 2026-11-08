"""Real-coded GA that minimises the turning-point fitness of one rule (rules.py) on a training window.

  initialisation  `population` individuals drawn uniformly inside the bounds; the rule's default parameters are always one of
                  them, so the result is never worse on the training window than the default.
  selection       tournament of 3.
  crossover       with probability `cx_prob` per pair; per gene, with probability 1/2, a BLX-0.5 blend of the two parents
                  (integer genes rounded afterwards).
  mutation        with probability `mut_prob` per gene: add N(0, (0.1 * (high - low))^2).
  repair          Rule.repair after every change: clip to the bounds, round integers, keep window orders.
  elitism         the best `elite` individuals go to the next generation unchanged.
The fitness of a parameter set is cached, so repeats (common once the population converges) cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from legacy_v1.ga.rules import Rule


@dataclass(frozen=True)
class GAConfig:
    population: int = 30
    generations: int = 20
    cx_prob: float = 0.7
    mut_prob: float = 0.2
    tournament: int = 3
    elite: int = 2
    seed: int = 0


def optimize(rule: Rule, objective: Callable[[dict], float], config: GAConfig = GAConfig()) -> dict:
    """Minimise `objective(params_dict)` over the rule's parameter space. Returns the best parameters, their objective value,
    the default's value, the per-generation history and the number of distinct parameter sets evaluated."""
    rng = np.random.default_rng(config.seed)
    low = np.array([p.low for p in rule.params], dtype=float)
    high = np.array([p.high for p in rule.params], dtype=float)
    span = high - low
    cache: dict[tuple, float] = {}

    def score(genes: Sequence[float]) -> float:
        key = tuple(rule.repair(genes))
        if key not in cache:
            value = objective(rule.as_dict(key))
            cache[key] = value if np.isfinite(value) else np.inf
        return cache[key]

    pop = [rule.repair(rule.default())] + [rule.repair(rng.uniform(low, high)) for _ in range(config.population - 1)]
    fit = [score(g) for g in pop]
    history = [{"generation": 0, "best": float(min(fit)), "mean": float(np.mean([f for f in fit if np.isfinite(f)] or [np.nan]))}]

    for gen in range(1, config.generations + 1):
        order = np.argsort(fit)
        children = [list(pop[i]) for i in order[:config.elite]]

        def pick() -> list[float]:
            contenders = rng.choice(len(pop), size=config.tournament, replace=False)
            return list(pop[min(contenders, key=lambda i: fit[i])])

        while len(children) < config.population:
            a, b = np.array(pick()), np.array(pick())
            if rng.random() < config.cx_prob:
                gamma = rng.uniform(-0.5, 1.5, size=len(a))
                mix = rng.random(len(a)) < 0.5
                a, b = np.where(mix, (1 - gamma) * a + gamma * b, a), np.where(mix, gamma * a + (1 - gamma) * b, b)
            for child in (a, b):
                mutate = rng.random(len(child)) < config.mut_prob
                child = child + mutate * rng.normal(0.0, 0.1 * span)
                children.append(rule.repair(child))
        pop = children[:config.population]
        fit = [score(g) for g in pop]
        finite = [f for f in fit if np.isfinite(f)]
        history.append({"generation": gen, "best": float(min(fit)), "mean": float(np.mean(finite)) if finite else float("nan")})

    best = int(np.argmin(fit))
    return {"params": rule.as_dict(pop[best]), "objective": float(fit[best]), "default_objective": score(rule.default()),
            "history": history, "evaluations": len(cache)}
