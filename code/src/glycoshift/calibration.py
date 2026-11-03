from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_temperature(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 10.0,
    candidates: int = 401,
) -> float:
    labels = np.asarray(labels, dtype=float)
    logits = np.asarray(logits, dtype=float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 1.0
    temperatures = np.geomspace(minimum, maximum, candidates)
    losses: list[float] = []
    for temperature in temperatures:
        probabilities = sigmoid(logits / temperature)
        probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
        loss = -np.mean(
            labels * np.log(probabilities)
            + (1 - labels) * np.log(1 - probabilities)
        )
        losses.append(float(loss))
    return float(temperatures[int(np.argmin(losses))])


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return sigmoid(np.asarray(logits, dtype=float) / temperature)


def fit_conformal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    alpha: float = 0.1,
) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0:
        raise ValueError("conformal calibration requires observations")
    true_class_probability = np.where(labels == 1, probabilities, 1 - probabilities)
    scores = 1 - true_class_probability
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    rank = min(max(rank, 1), len(scores))
    return float(np.partition(scores, rank - 1)[rank - 1])


def conformal_prediction_set(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    include_negative = (1 - probabilities) >= 1 - threshold
    include_positive = probabilities >= 1 - threshold
    return np.stack((include_negative, include_positive), axis=1)
