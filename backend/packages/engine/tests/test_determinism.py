import hashlib

from returniq_engine import generate_dataset
from returniq_engine.export_fixtures import build_fixtures
from returniq_engine.generator import to_csv_bytes


def _hashes(g):
    return [hashlib.sha256(to_csv_bytes(f)).hexdigest() for f in (g.orders, g.shipments, g.returns)]


def test_same_seed_identical_csv_hashes():
    assert _hashes(generate_dataset(1200, 42)) == _hashes(generate_dataset(1200, 42))


def test_different_seed_differs():
    assert _hashes(generate_dataset(1200, 42)) != _hashes(generate_dataset(1200, 43))


def test_fixture_export_is_byte_identical():
    a = build_fixtures(42, 400, include_model=False)
    b = build_fixtures(42, 400, include_model=False)
    assert a.keys() == b.keys()
    assert all(a[k] == b[k] for k in a)


def test_assessments_deterministic_ignoring_latency(eng, ds42, settings):
    from returniq_contracts import Stage

    a = eng.assess_batch(ds42, Stage.PRE_DISPATCH, settings, None)
    b = eng.assess_batch(ds42, Stage.PRE_DISPATCH, settings, None)
    assert [x.model_dump(exclude={"latency_ms"}) for x in a] == [
        x.model_dump(exclude={"latency_ms"}) for x in b
    ]
