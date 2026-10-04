"""Shared fixtures: seed-42 dataset is generated and prepared once per session."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from returniq_contracts import Settings, Stage

from returniq_engine import Dataset, ReturnIQEngine, default_settings, generate_dataset

FAR_FUTURE = datetime(2027, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="session")
def eng() -> ReturnIQEngine:
    return ReturnIQEngine()


@pytest.fixture(scope="session")
def settings() -> Settings:
    return default_settings()


@pytest.fixture(scope="session")
def g42():
    return generate_dataset(1200, 42)


@pytest.fixture(scope="session")
def ds42(g42) -> Dataset:
    ds = Dataset.from_generated(g42)
    ds.prepare()
    return ds


@pytest.fixture(scope="session")
def model_s2(eng, ds42):
    return eng.train(ds42, Stage.POST_DELIVERY, None)


@pytest.fixture(scope="session")
def model_s1(eng, ds42):
    return eng.train(ds42, Stage.PRE_DISPATCH, None)
