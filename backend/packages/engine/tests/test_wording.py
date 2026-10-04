import pathlib

from returniq_contracts import Stage

from returniq_engine.explain import FEATURE_LABELS

BANNED = ("fraud", "fake customer")
ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_no_banned_words_in_emitted_assessment_text(eng, ds42, settings, model_s2):
    for stage, model in ((Stage.PRE_DISPATCH, None), (Stage.POST_DELIVERY, model_s2)):
        for a in eng.assess_batch(ds42, stage, settings, model):
            blob = (a.reason + " ".join(s.label + s.evidence for s in a.signals)).lower()
            assert not any(b in blob for b in BANNED)


def test_feature_labels_clean():
    assert not any(b in v.lower() for v in FEATURE_LABELS.values() for b in BANNED)


def test_sources_and_docs_never_use_banned_words():
    files = list((ROOT / "packages/engine/returniq_engine").glob("*.py"))
    files += [p for p in (ROOT / "docs").glob("*.md")]
    for f in files:
        assert not any(b in f.read_text().lower() for b in BANNED), f
