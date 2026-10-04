from returniq_engine import report


def test_report_runs_and_is_deterministic(capsys):
    report.main(["--seed", "42", "--n", "600", "--no-model"])
    first = capsys.readouterr().out
    report.main(["--seed", "42", "--n", "600", "--no-model"])
    second = capsys.readouterr().out
    assert first == second
    assert (
        "band table [rules-only]" in first and "simulated, on planted synthetic patterns" in first
    )
    assert "INSUFFICIENT_DATA" in first


def test_report_with_model(capsys):
    report.main(["--seed", "42", "--n", "1200"])
    out = capsys.readouterr().out
    assert "ML test metrics" in out and "rules+ml" in out
