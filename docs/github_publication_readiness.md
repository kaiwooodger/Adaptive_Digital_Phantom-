# GitHub Publication Readiness

Date: 2026-06-25

## Current Release Position

This repository is ready to become a GitHub **research demonstrator source repo** after a clean Git initialization and CI pass. It is not ready to publish as a full dataset or clinical product.

The active product baseline is:

- Case ID: `btcv_stage007_active_full`
- Status: `research_demo_ready`
- QA gate: `22 pass / 0 review / 0 fail`
- Current 3D render: `outputs/product_cases/btcv_stage007_active_full/render3d/btcv_stage007_active_full_vascular_network_3d_render_preview_v001.png`
- Vessel-visible render: `outputs/product_cases/btcv_stage007_active_full/render3d/btcv_stage007_active_full_vessel_visible_3d_render_preview_v001.png`

## What Can Be Published

Commit these source-controlled assets:

- `src/`
- `tests/`
- `configs/`
- `docs/`
- `scripts/`
- `README.md`
- `pyproject.toml`
- `environment.yml`
- `.gitignore`

Keep these out of GitHub:

- `data/`
- `RawData/`
- `outputs/`
- `.deps/`
- `.tools/`
- `.tmp/`
- medical images, DICOMs, generated NIfTI volumes, generated meshes, and release archives

## GI Module Status

The code supports a full GI organ module with:

- Preserved BTCV stomach/esophagus/pancreas/spleen/gallbladder/adrenal labels.
- Generic stomach/GI gas-fluid lumen layer.
- Explicit placeholder labels for duodenum, small bowel, colon, rectum, and specific GI lumen compartments.
- A real-mask replacement bridge for co-registered `stomach`, `duodenum`, `small_bowel`, `colon`, and `rectum` segmentations.

The active rendered baseline still uses the generic GI lumen. Real bowel/colon/small-intestine masks are not staged yet. The Docker TotalSegmentator path is prepared, but it is blocked locally until more disk space is available.

## Required Before Public Release

1. Keep the repository private until the first GitHub Actions CI pass completes.
2. Choose an open-source license before making the repository public. The current `NOTICE.md` is all-rights-reserved and intentionally conservative.
3. Confirm no patient-identifiable or dataset-restricted files are staged.
4. Add a short limitations section to the GitHub release notes: research only, not clinical, not diagnostic, not treatment planning, synthetic/placeholder GI and vascular elements remain.
5. If publishing generated figures or artifacts, verify every source dataset permits redistribution.

## Suggested First Push

```bash
git init
git status --ignored
git add .gitignore README.md NOTICE.md pyproject.toml environment.yml Makefile .github configs docs scripts src tests data outputs
git status --short
git commit -m "Initial research demonstrator source release"
```

Do not use `git add .` for the first commit.
