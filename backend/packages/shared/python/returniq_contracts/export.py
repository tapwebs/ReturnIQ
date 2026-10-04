"""Generate `packages/shared/schemas/*.json` and `CONTRACT_VERSION` from the Pydantic models.

Usage: ``python -m returniq_contracts.export [--out DIR]``. Output is deterministic
(sorted keys, 2-space indent, trailing newline). Never edit the generated files by hand.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from pydantic import BaseModel

from . import CONTRACT_VERSION, models

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "schemas"


def exported_models() -> dict[str, type[BaseModel]]:
    """All public contract models defined in ``models.py`` keyed by class name."""
    found: dict[str, type[BaseModel]] = {}
    for name, obj in sorted(vars(models).items()):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj.__module__ == models.__name__
            and name != "Contract"
        ):
            found[name] = obj
    return found


def render_schema(model: type[BaseModel]) -> str:
    """Deterministic JSON text for one model's schema."""
    schema = model.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def export(out: Path = DEFAULT_OUT) -> list[Path]:
    """Write every schema plus CONTRACT_VERSION; return the written paths."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in exported_models().items():
        path = out / f"{name}.json"
        path.write_text(render_schema(model), encoding="utf-8")
        written.append(path)
    version_path = out.parent / "CONTRACT_VERSION"
    version_path.write_text(CONTRACT_VERSION + "\n", encoding="utf-8")
    written.append(version_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    paths = export(args.out)
    print(f"wrote {len(paths)} files to {args.out} (contract {CONTRACT_VERSION})")


if __name__ == "__main__":
    main()
