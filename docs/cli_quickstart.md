# CLI Quickstart

The command-line interface is exposed as `phantom-twin` after an editable install.
Run commands from a cloned checkout so the default `configs/` paths resolve
exactly as they do in CI.

## Setup

```bash
cd phantom-digital-twin
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

For medical-image workflows and the full test suite:

```bash
.venv/bin/pip install -e ".[dev,medical]"
```

Or use the helper:

```bash
scripts/bootstrap_env.sh python3.12
```

## Smoke Test

```bash
scripts/smoke_test_cli.sh .venv/bin/python
```

Expected result:

```text
CLI smoke tests passed.
```

## Core Commands

```bash
phantom-twin phase1-summary
phantom-twin materials-check
phantom-twin datasets-list
phantom-twin inspect-nifti data/raw/example.nii.gz
```

## Product-Case Entry Points

Use these after staging local CT/segmentation inputs. The repository does not
include medical images or generated NIfTI/STL artifacts.

```bash
phantom-twin build-patient-phantom-adapter --help
phantom-twin run-patient-case-adapter --help
phantom-twin build-product-case --help
phantom-twin build-product-release-case --help
```

## Main Research Modules

The CLI currently covers these major workflows:

- Anatomy/material maps: `build-digital-torso`, `build-combined-digital-phantom`.
- Anthropometry: `build-anthropometric-torso-morph`, `build-profile-sweep`, `build-profile-envelope`.
- Population/PCA morphology: `build-population-cohort`, `generate-pca-mode-variants`, `qa-pca-modes`.
- Vascular graph and labels: `build-vascular-network-scaffold`, `build-labeled-vessel-vascular-graph`, `build-label-vessel-flow-domain`.
- Flow: `build-flow-boundary-package`, `build-flow-1d-model`, `build-pulsatile-flow-model`, `build-coupled-pulsatile-flow-model`.
- Radiotherapy QA: `build-radiotherapy-qa-package`, `build-rt-planning-bundle`, `build-dose-gamma-qa`.
- Rendering: `render-combined-3d`, `render-vascular-network-3d`, `render-3d-atlas`.

Run any command with `--help` for full arguments.
