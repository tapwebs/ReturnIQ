"""IsolationForest anomaly component, percentile-mapped within the seller's own history."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

N_TREES = 100
RANDOM_STATE = 42


def anomaly_percentiles(frame: pd.DataFrame) -> np.ndarray:
    """0 = most normal, 100 = most unusual, ranked against this dataset's own rows."""
    x = frame.fillna(frame.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    if len(x) < 2:
        return np.zeros(len(x))
    forest = IsolationForest(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=1).fit(x)
    unusual = -forest.score_samples(x)
    return pd.Series(unusual).rank(method="average", pct=True).to_numpy() * 100.0
