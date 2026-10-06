"""Build <team>_submission.zip in the exact layout required by the problem statement.

<team>_submission.zip
|-- output/matching_results.tsv, output/candidate_pairs.tsv
|-- code/business_entity_resolution/{src/, README.md, requirements.txt}
`-- Documentation_template.md

Before zipping it re-runs the submission checks on the output files and refuses to build if
anything is missing or invalid. Caches, artifacts, data, virtual envs and stray test files in
src/ are never included.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from .config import CFG, PROJECT_ROOT, SUBMISSION_ROOT, Config
from .submission import check_submission

LOG = logging.getLogger("ber.package")

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "artifacts", "output", ".venv", "venv"}


def _source_files(include_tests: bool) -> list[tuple[Path, str]]:
    files = []
    for p in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        if EXCLUDE_DIRS & set(p.parts) or p.name.startswith("test_"):
            if p.name.startswith("test_"):
                LOG.warning("skipping stray test file inside src/: %s", p.name)
            continue
        files.append((p, f"code/business_entity_resolution/src/{p.relative_to(PROJECT_ROOT / 'src').as_posix()}"))
    for name in ("README.md", "requirements.txt"):
        files.append((PROJECT_ROOT / name, f"code/business_entity_resolution/{name}"))
    if include_tests:
        files.append((PROJECT_ROOT / "pytest.ini", "code/business_entity_resolution/pytest.ini"))
        for p in sorted((PROJECT_ROOT / "tests").glob("*.py")):
            files.append((p, f"code/business_entity_resolution/tests/{p.name}"))
    return files


def build_package(team: str, cfg: Config = CFG, include_tests: bool = False) -> Path:
    out_dir = Path(cfg.output_dir)
    match, cand = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    doc = SUBMISSION_ROOT / "Documentation_template.md"
    files = _source_files(include_tests) + [(match, "output/matching_results.tsv"),
                                            (cand, "output/candidate_pairs.tsv"),
                                            (doc, "Documentation_template.md")]
    missing = [str(p) for p, _ in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("cannot package, missing:\n  " + "\n  ".join(missing))
    issues = check_submission(match, cand, "test", cfg)
    if issues:
        raise ValueError("output files fail the submission checks:\n  " + "\n  ".join(issues))
    zip_path = SUBMISSION_ROOT / f"{team}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            zf.write(src, arc)
    LOG.info("wrote %s (%d files, %.0f MB)", zip_path, len(files), zip_path.stat().st_size / 2**20)
    for _, arc in files:
        LOG.info("  %s", arc)
    return zip_path
