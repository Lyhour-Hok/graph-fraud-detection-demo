"""Paths and small helpers shared by the offline pipeline stages."""
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"      # shipped with the website
BUILD_DIR = PROJECT_DIR / "build"              # offline-only intermediates (not deployed)
MANIFEST = ARTIFACTS_DIR / "manifest.json"

ARTIFACTS_DIR.mkdir(exist_ok=True)
BUILD_DIR.mkdir(exist_ok=True)

GRAPHS = ["bipartite", "tripartite"]
OPERATORS = ["hadamard", "average", "l1", "l2"]


def default_train_csv():
    return os.environ.get("FRAUD_TRAIN_CSV", str(PROJECT_DIR / "data" / "fraudTrain.csv"))


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def update_manifest(path, value):
    """Set manifest[path[0]][path[1]]... = value and write it back."""
    m = load_manifest()
    node = m
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    m["updated"] = datetime.now().isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")


def check(label, got, expected):
    """Print a got-vs-thesis comparison line."""
    mark = "OK " if got == expected else "DIFF"
    print(f"  [{mark}] {label}: {got:,} (thesis: {expected:,})")
