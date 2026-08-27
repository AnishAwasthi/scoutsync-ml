"""
Guards against presenting invented numbers as model output.

A previous release shipped a Streamlit dashboard that inserted hard-coded projections
for three real major league players and rendered them under headings like "MLB Projection
Summary" and "SHAP Explainability". No model ran. The repository was public and linked
from a resume.

These tests are cheap and deliberately blunt. They exist so that failure mode cannot
quietly return.
"""

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Real players whose names appearing next to a projection value would mean fabricated
# output has come back.
REAL_PLAYER_NAMES = ["Ohtani", "Judge", "Skenes", "Trout", "Betts", "Soto"]

PROJECTION_FIELDS = [
    "proj_wOBA",
    "proj_ERA",
    "proj_wOBA_lower_90",
    "proj_wOBA_upper_90",
    "proj_ERA_lower_90",
    "proj_ERA_upper_90",
]

SOURCE_FILES = sorted(
    [PROJECT_ROOT / "dashboard.py"]
    + [p for p in (PROJECT_ROOT / "src").rglob("*.py")]
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_real_player_names_in_application_source():
    offenders = []
    for path in SOURCE_FILES:
        text = _read(path)
        for name in REAL_PLAYER_NAMES:
            # Ignore the explanatory docstrings that describe the removed behaviour.
            for line in text.splitlines():
                stripped = line.strip()
                if name in line and not stripped.startswith("#"):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {stripped[:90]}")
    assert not offenders, "real player names found in source:\n" + "\n".join(offenders)


def test_dashboard_never_constructs_a_projection_with_literal_values():
    """
    Any ``MlbProjection(...)`` built in the dashboard with numeric literals for a
    projection field is fabricated output by definition -- projections must come from
    the model, through the pipeline.
    """
    tree = ast.parse(_read(PROJECT_ROOT / "dashboard.py"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "MlbProjection":
            continue
        for keyword in node.keywords:
            if keyword.arg in PROJECTION_FIELDS and isinstance(keyword.value, ast.Constant):
                offenders.append(f"{keyword.arg}={keyword.value.value}")

    assert not offenders, (
        "dashboard.py builds projections from literals: " + ", ".join(offenders)
    )


def test_no_hardcoded_shap_dictionaries_in_the_dashboard():
    """SHAP payloads must be computed, never typed out as a literal mapping."""
    text = _read(PROJECT_ROOT / "dashboard.py")
    suspicious = re.findall(r'"[A-Z][A-Za-z ()\-\.]{6,}"\s*:\s*-?\d*\.\d+', text)
    assert not suspicious, f"hard-coded SHAP-like literals in dashboard.py: {suspicious}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.name))
def test_source_files_parse(path):
    """A syntax error in any shipped module should fail CI, not the hosted demo."""
    ast.parse(_read(path))
