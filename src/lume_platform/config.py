"""Central configuration: paths, Spark tuning, artifact locations."""

from __future__ import annotations

import os
from pathlib import Path


def _default_project_root() -> Path:
    # src/lume_platform/config.py -> parents[2] == repo root (BigData)
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = Path(os.environ.get("LUME_PROJECT_ROOT", _default_project_root()))
DATA_ROOT: Path = Path(os.environ.get("LUME_DATA_ROOT", PROJECT_ROOT / "datasets"))
ARTIFACTS_DIR: Path = Path(os.environ.get("LUME_ARTIFACTS", PROJECT_ROOT / "artifacts"))
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
CLEANED_DIR: Path = ARTIFACTS_DIR / "cleaned_parquet"
EXPORT_DIR: Path = PROJECT_ROOT / "output_production_final"
TABLEAU_EXPORT_DIR: Path = PROJECT_ROOT / "tableau_exports"
EVAL_DIR: Path = PROJECT_ROOT / "model_evaluations"

# Default public Tableau embed (replace after publishing your workbook)
TABLEAU_PUBLIC_EMBED_URL: str = os.environ.get(
    "LUME_TABLEAU_EMBED_URL",
    "https://public.tableau.com/views/Placeholder/Sheet1?:showVizHome=no&:embed=y",
)

SPARK_SHUFFLE_PARTITIONS: int = int(os.environ.get("LUME_SPARK_SHUFFLE_PARTITIONS", "200"))
SPARK_DRIVER_MEMORY: str = os.environ.get("LUME_SPARK_DRIVER_MEMORY", "8g")


def ensure_dirs() -> None:
    for p in (MODELS_DIR, CLEANED_DIR, EXPORT_DIR, TABLEAU_EXPORT_DIR, EVAL_DIR):
        p.mkdir(parents=True, exist_ok=True)
