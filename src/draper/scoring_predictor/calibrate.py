"""Per-head isotonic calibration for the scoring predictor.

Trains one isotonic regressor per head on the validation split (``y_pred``
from the model → ``y_true`` from the v3 corpus), then applies it at inference
time to map raw model output → calibrated [0, 1].

We use isotonic over Platt scaling because (a) it doesn't assume a logistic
shape, (b) it handles the long-tailed v3 distribution better, and (c) one
fit per head means four small regressors that take milliseconds to apply.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from draper.scoring_predictor.data import HEAD_NAMES


class HeadCalibrators:
    """Fitted isotonic calibrators for each of the four heads.

    Persists to disk as a JSON file storing each head's monotone breakpoints
    so we don't have to pickle scikit-learn objects (their pickle format isn't
    a stable cross-version contract).
    """

    def __init__(self, calibrators: dict[str, IsotonicRegression]) -> None:
        self._calibrators = calibrators

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Apply per-head calibration to a ``(N, H)`` prediction array."""
        out = np.empty_like(predictions, dtype=float)
        for h, name in enumerate(HEAD_NAMES):
            cal = self._calibrators.get(name)
            if cal is None:
                out[:, h] = predictions[:, h]
            else:
                out[:, h] = cal.predict(predictions[:, h])
        return np.clip(out, 0.0, 1.0)

    def transform_one(self, prediction_row: dict[str, float]) -> dict[str, float]:
        """Apply calibration to a single prediction dict."""
        out: dict[str, float] = {}
        for name in HEAD_NAMES:
            raw = float(prediction_row.get(name, 0.0))
            cal = self._calibrators.get(name)
            if cal is None:
                out[name] = raw
            else:
                out[name] = float(np.clip(cal.predict(np.array([raw]))[0], 0.0, 1.0))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, dict[str, list[float]]] = {}
        for name, cal in self._calibrators.items():
            payload[name] = {
                "x": list(map(float, cal.X_thresholds_)),
                "y": list(map(float, cal.y_thresholds_)),
            }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> HeadCalibrators:
        path = Path(path)
        payload: dict[str, dict[str, list[float]]] = json.loads(path.read_text())
        calibrators: dict[str, IsotonicRegression] = {}
        for name, table in payload.items():
            x = np.asarray(table["x"], dtype=float)
            y = np.asarray(table["y"], dtype=float)
            # Reconstruct by re-fitting to stored breakpoints. This faithfully
            # reproduces the curve because IsotonicRegression's output at the
            # fit points equals the input y values by definition. Verify this
            # assumption holds: the length of thresholds equals the fit size.
            if len(x) < 2:
                # Skip heads with fewer than 2 breakpoints — isotonic would
                # return a constant, not useful.
                continue
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(x, y)
            # Defensive: after fit, the stored x/y should be recoverable as the
            # internal thresholds (or close, within floating-point error).
            # This catches silent corruption in the JSON file.
            if (
                len(cal.X_thresholds_) != len(x)
                or not np.allclose(cal.X_thresholds_, x, rtol=1e-10)
                or not np.allclose(cal.y_thresholds_, y, rtol=1e-10)
            ):
                raise ValueError(
                    f"Isotonic calibrator {name} failed round-trip verification. "
                    f"Fit produced {len(cal.X_thresholds_)} thresholds, "
                    f"expected {len(x)}. Likely corrupted calibrators file."
                )
            calibrators[name] = cal
        return cls(calibrators)


def fit_calibrators(
    raw_predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
) -> HeadCalibrators:
    """Fit one isotonic calibrator per head.

    Heads with fewer than 10 valid (mask=True) rows skip calibration — too
    little data to fit a useful curve, and the raw sigmoid output is bounded
    to [0, 1] anyway.
    """
    calibrators: dict[str, IsotonicRegression] = {}
    for h, name in enumerate(HEAD_NAMES):
        mask = target_mask[:, h].astype(bool)
        if int(mask.sum()) < 10:
            continue
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(raw_predictions[mask, h], targets[mask, h])
        calibrators[name] = cal
    return HeadCalibrators(calibrators)


def calibrate_array(
    calibrators: HeadCalibrators,
    rows: Sequence[Sequence[float]],
) -> list[dict[str, float]]:
    """Apply calibrators to a sequence of raw 4-tuples and return dicts."""
    arr = np.asarray(rows, dtype=float)
    cal = calibrators.transform(arr)
    return [
        {name: float(val) for name, val in zip(HEAD_NAMES, row, strict=True)}
        for row in cal.tolist()
    ]
