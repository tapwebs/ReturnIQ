"""ReturnIQ risk engine: pure Python, deterministic, no I/O."""

from .api import Engine, ReturnIQEngine
from .generator import GeneratedDataset, generate_dataset
from .prepare import Dataset
from .settings import ENGINE_VERSION, RULE_VERSION, default_settings

__version__ = ENGINE_VERSION
__all__ = [
    "Dataset",
    "Engine",
    "GeneratedDataset",
    "ReturnIQEngine",
    "RULE_VERSION",
    "default_settings",
    "generate_dataset",
    "__version__",
]
