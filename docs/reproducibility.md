# Reproducibility Guide

This repository is source-code reproducible. It intentionally does not ship
medical image volumes, patient data, generated NIfTI maps, generated meshes, or
large release artifacts.

## What Is Reproducible From GitHub

From a clean clone, users can reproduce:

- Python package installation.
- CLI command discovery and smoke tests.
- Unit tests using synthetic/minimal fixtures.
- Dataset staging manifests and documented commands.
- Product/release package generation once users provide compatible local data.

## What Must Be Supplied Locally

Users must stage their own data according to dataset licenses and institutional
approval:

- CT/CTA/CTV NIfTI or DICOM-derived volumes.
- Organ segmentations.
- Vessel segmentations or CTA-derived vessel masks.
- Optional GI masks from a supported segmentation source.

Generated files belong in ignored folders:

```text
data/raw/
data/interim/
data/processed/
data/derived/
data/validation/
outputs/
```

## Clean Clone Verification

```bash
git clone <repo-url>
cd phantom-digital-twin
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e ".[dev,medical]"
scripts/smoke_test_cli.sh .venv/bin/python
.venv/bin/python -m pytest -q
```

GitHub Actions runs the same smoke/test pathway on Python 3.10 and 3.12.

## Research-Only Limitation

The current phantom is a research/engineering demonstrator. It is not a
clinical device, treatment-planning system, diagnostic tool, or validated
patient-specific digital twin.
