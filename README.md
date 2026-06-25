# Phantom Digital Twin

Research scaffold for a modular, lifelike anthropomorphic torso phantom with a
matched computational twin. Phase 1 targets CT realism, radiotherapy QA, and
large-vessel vascular flow.

This project is for research and engineering development only. It is not a
clinical device, treatment-planning system, diagnostic tool, or regulatory
submission package.

## Phase 1 Scope

The first build is a thorax/upper-abdomen MVP rather than a full body. The goal
is to prove the full loop:

```text
public CT/CTA reference data
  -> segmentation and labels
  -> density/material map
  -> mesh/CAD assets
  -> physical phantom modules
  -> CT/radiotherapy/flow validation
  -> update the digital twin
```

Initial tracks:

- CT: anatomy, HU targets, tissue-equivalent material coupons, scan validation.
- Radiotherapy: relative electron-density targets, dose-grid comparison, gamma
  analysis using PyMedPhys once dose data exists.
- Vascular flow: aorta/major-vessel geometry, pump-loop targets, CFD boundary
  conditions, flow QA.

## Dataset Starting Point

We will use a two-stream data strategy:

- Immediate anatomy stream: CT-ORG, because it provides CT volumes and organ
  masks in NIfTI format under an open license.
- Vascular stream: public CTA/aortic datasets such as ImageTBAD or the type-B
  aortic dissection Figshare collection for large-vessel geometry.
- Whole-body upgrade path: Healthy-Total-Body-CTs, XCAT, or ICRP-style reference
  models once licensing/access is sorted.

See [docs/data_sources.md](docs/data_sources.md).

## Repository Layout

```text
configs/
  datasets.yaml              dataset manifest and source choices
  materials.yaml             seed HU/density/RED material targets
  phase1_torso_mvp.yaml      build scope, modules, milestones
  labelmaps/                 dataset label conventions
data/
  raw/                       downloaded source data, ignored by git
  interim/                   converted/reoriented data
  processed/                 digital-twin working products
outputs/
  meshes/                    STL/OBJ/PLY exports
  cad/                       CAD-ready exchange files
  reports/                   validation summaries
src/phantom_twin/            lightweight pipeline helpers
tests/                       config and manifest checks
```

## Quick Start

Clone the repository and create a Python 3.10+ environment:

```bash
git clone <repo-url>
cd phantom-digital-twin
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

Run the CLI smoke test:

```bash
scripts/smoke_test_cli.sh .venv/bin/python
```

Then try the core source-only commands:

```bash
phantom-twin phase1-summary
phantom-twin materials-check
phantom-twin datasets-list
```

Optional medical-imaging tools are only needed once data has been downloaded:

```bash
.venv/bin/pip install -e ".[dev,medical]"
```

See [docs/cli_quickstart.md](docs/cli_quickstart.md) and
[docs/reproducibility.md](docs/reproducibility.md).

Example mesh export after CT-ORG files are available:

```bash
phantom-twin export-label-mesh \
  --labels data/raw/ct_org/labels/labels-0.nii.gz \
  --label-id 3 \
  --output outputs/meshes/ct_org_case0_lungs.stl
```

## First Engineering Milestone

Before printing an organ, build and scan material coupons. A beautiful phantom
with wrong HU/RED values is a sculpture; a calibrated material coupon library is
the start of a test instrument.

## Current Staged Data

Phase 1 data stage 001 has been completed. One CT-ORG anatomy case and one
ImageTBAD vascular CTA case are staged locally, inspected, and exported to STL.

See [docs/data_stage_001.md](docs/data_stage_001.md).

Run mesh QA with:

```bash
phantom-twin mesh-qa \
  outputs/meshes/*.stl \
  --output outputs/reports/mesh_qa_stage001.md
```

Clean meshes for CAD preparation with:

```bash
phantom-twin clean-meshes \
  outputs/meshes/*.stl \
  --output-dir outputs/cad/cleaned_meshes \
  --report outputs/reports/mesh_cleaning_stage001.md \
  --qa-report outputs/reports/mesh_qa_cleaned_stage001.md
```

Prepare the first vascular flow-module geometry with:

```bash
phantom-twin prepare-vascular-module \
  --labels data/raw/imagetbad/case125/125_label.nii.gz \
  --label-id 1 \
  --case-id imagetbad_case125_true_lumen \
  --output-dir outputs/cad/vascular_module \
  --centerline-method skeleton \
  --report outputs/reports/vascular_module_stage001.md \
  --qa-report outputs/reports/vascular_module_mesh_qa_stage001.md
```

Create the first port-adapter flow-loop reference assembly with:

```bash
phantom-twin design-flow-loop \
  --ports outputs/cad/vascular_module/imagetbad_case125_true_lumen_ports_v001.yaml \
  --lumen-mesh outputs/cad/vascular_module/imagetbad_case125_true_lumen_smoothed_lumen_v001.stl \
  --output-dir outputs/cad/vascular_flow_loop \
  --report outputs/reports/vascular_flow_loop_stage001.md \
  --qa-report outputs/reports/vascular_flow_loop_mesh_qa_stage001.md
```

Build the first watertight printable vascular flow cartridge with:

```bash
phantom-twin build-printable-cartridge \
  --labels data/raw/imagetbad/case125/125_label.nii.gz \
  --label-id 1 \
  --flow-loop-spec outputs/cad/vascular_flow_loop/imagetbad_case125_true_lumen_flow_loop_spec_v001.yaml \
  --centerline-csv outputs/cad/vascular_module/imagetbad_case125_true_lumen_centerline_v001.csv \
  --output-dir outputs/cad/vascular_cartridge \
  --report outputs/reports/vascular_printable_cartridge_stage001.md \
  --qa-report outputs/reports/vascular_printable_cartridge_mesh_qa_stage001.md
```

Build the first digital-only torso material/density phantom volume with:

```bash
phantom-twin build-digital-torso \
  --ct data/raw/ct_org/volumes/volume-0.nii.gz \
  --labels data/raw/ct_org/labels/labels-0.nii.gz \
  --labelmap configs/labelmaps/ct_org.yaml \
  --materials configs/materials.yaml \
  --case-id ct_org_case0 \
  --output-dir outputs/digital/torso \
  --report outputs/reports/digital_torso_stage001.md
```

Build the combined digital torso + vascular phantom package with:

```bash
phantom-twin build-combined-digital-phantom \
  --torso-material-labels outputs/digital/torso/ct_org_case0_torso_material_labels_v001.nii.gz \
  --torso-body-mask outputs/digital/torso/ct_org_case0_torso_body_mask_v001.nii.gz \
  --source-ct data/raw/ct_org/volumes/volume-0.nii.gz \
  --vascular-labels data/raw/imagetbad/case125/125_label.nii.gz \
  --vascular-label-id 1 \
  --flow-loop-spec outputs/cad/vascular_flow_loop/imagetbad_case125_true_lumen_flow_loop_spec_v001.yaml \
  --materials configs/materials.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/digital/combined \
  --report outputs/reports/combined_digital_phantom_stage001.md
```

Create a BMI/waist/height-adapted anthropometric torso variant:

```bash
phantom-twin build-anthropometric-torso-morph \
  --combined-spec outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125_bmi32_waist110 \
  --output-dir outputs/digital/anthropometric_morph \
  --target-height-cm 175.0 \
  --target-bmi 32.0 \
  --target-waist-cm 110.0 \
  --baseline-height-cm 170.0 \
  --baseline-bmi 24.0 \
  --report outputs/reports/anthropometric_torso_morph_stage001.md
```

Build a registration/PCA-based anatomy variant from staged population material-label segmentations:

```bash
phantom-twin build-statistical-anatomy-morph \
  --combined-spec outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml \
  --population-labels \
    outputs/digital/torso/ct_org_case0_torso_material_labels_v001.nii.gz \
  --population-case-ids ct_org_case0 \
  --case-id ct_org_case0_statistical_bootstrap_bmi29_waist104 \
  --output-dir outputs/digital/statistical_anatomy_morph \
  --target-height-cm 176.0 \
  --target-bmi 29.0 \
  --target-waist-cm 104.0 \
  --baseline-height-cm 170.0 \
  --baseline-bmi 24.0 \
  --report outputs/reports/statistical_anatomy_morph_stage001.md
```

For real population statistics, process each additional CT segmentation through `build-digital-torso` first, then pass all resulting material-label NIfTIs to `--population-labels`. PCA modes become meaningful once several independent cases are staged.

Build a population cohort manifest, registration QA, per-case previews, and PCA-ready inputs with:

```bash
phantom-twin stage-ct-org-label-cohort \
  --case-indices 0 1 2 3 4 5 6 7 8 9 \
  --output-dir data/processed/ct_org_label_population \
  --report outputs/reports/ct_org_label_population_stage001.md

phantom-twin build-population-cohort \
  --combined-spec outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml \
  --population-labels \
    data/processed/ct_org_label_population/ct_org_case0_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case1_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case2_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case3_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case4_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case5_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case6_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case7_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case8_label_only_material_labels_v001.nii.gz \
    data/processed/ct_org_label_population/ct_org_case9_label_only_material_labels_v001.nii.gz \
  --population-case-ids ct_org_case0 ct_org_case1 ct_org_case2 ct_org_case3 ct_org_case4 ct_org_case5 ct_org_case6 ct_org_case7 ct_org_case8 ct_org_case9 \
  --cohort-id ct_org_population_stage001 \
  --output-dir outputs/digital/population_cohort \
  --report outputs/reports/population_cohort_stage001.md
```

Generate a PCA anatomy mode comparison atlas from a cohort spec with:

```bash
phantom-twin generate-pca-mode-variants \
  --combined-spec outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml \
  --cohort-spec outputs/digital/population_cohort_ctorg_labels/ct_org_label_population8_stage001_population_cohort_spec_v001.yaml \
  --case-id ct_org_label_population8_pca_modes_stage001 \
  --output-dir outputs/digital/pca_mode_variants \
  --mode-count 3 \
  --amplitude 1.0 \
  --target-height-cm 176.0 \
  --target-bmi 29.0 \
  --target-waist-cm 104.0 \
  --baseline-height-cm 170.0 \
  --baseline-bmi 24.0 \
  --max-modes 6 \
  --report outputs/reports/pca_mode_variant_atlas_stage001.md
```

Rank and approve/reject the PCA modes from the atlas metrics with:

```bash
phantom-twin qa-pca-modes \
  --metrics-csv outputs/digital/pca_mode_variants/ct_org_label_population8_pca_modes_stage001_pca_mode_variant_metrics_v001.csv \
  --atlas-spec outputs/digital/pca_mode_variants/ct_org_label_population8_pca_modes_stage001_pca_mode_variant_atlas_spec_v001.yaml \
  --case-id ct_org_label_population8_pca_modes_stage001 \
  --output-dir outputs/digital/pca_mode_qa \
  --report outputs/reports/pca_mode_qa_stage001.md
```

Assemble a disk-light approved PCA phantom release set with:

```bash
phantom-twin build-approved-pca-phantom-set \
  --qa-decisions outputs/digital/pca_mode_qa/ct_org_label_population8_pca_modes_stage001_pca_mode_qa_decisions_v001.yaml \
  --atlas-spec outputs/digital/pca_mode_variants/ct_org_label_population8_pca_modes_stage001_pca_mode_variant_atlas_spec_v001.yaml \
  --case-id ct_org_label_population8_pca_modes_stage001 \
  --output-dir outputs/digital/approved_pca_phantom_set \
  --report outputs/reports/approved_pca_phantom_set_stage001.md
```

Run a disk-light anatomy + RT + flow experiment comparison across the approved set with:

```bash
phantom-twin run-phantom-experiment-set \
  --approved-set-manifest outputs/digital/approved_pca_phantom_set/ct_org_label_population8_pca_modes_stage001_approved_pca_phantom_set_manifest_v001.yaml \
  --rt-planning-spec outputs/radiotherapy/planning_bundle/ct_org_case0_imagetbad_case125_rt_planning_bundle_spec_v001.yaml \
  --dose-gamma-spec outputs/radiotherapy/dose_gamma_qa/ct_org_case0_imagetbad_case125_dose_gamma_qa_spec_v001.yaml \
  --flow-model-spec outputs/sim/flow_coupled_pulsatile/ct_org_case0_imagetbad_case125_coupled_pulsatile_flow_model_v001.yaml \
  --case-id ct_org_label_population8_pca_modes_stage001 \
  --output-dir outputs/experiments/approved_pca_phantom_set \
  --report outputs/reports/approved_pca_phantom_experiment_set_stage001.md
```

Create a variant-specific RT/flow rerun harness for the highest-impact PCA mode with:

```bash
phantom-twin build-variant-rerun-harness \
  --approved-set-manifest outputs/digital/approved_pca_phantom_set/ct_org_label_population8_pca_modes_stage001_approved_pca_phantom_set_manifest_v001.yaml \
  --variant-id mode01_neg \
  --baseline-combined-spec outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml \
  --flow-model-spec outputs/sim/flow_coupled_pulsatile/ct_org_case0_imagetbad_case125_coupled_pulsatile_flow_model_v001.yaml \
  --case-id ct_org_label_population8_pca_modes_stage001_mode01_neg \
  --output-dir outputs/experiments/variant_rerun_harness
```

Generate 3D renderable meshes and a transparent scene preview with:

```bash
phantom-twin render-combined-3d \
  --combined-labels outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_material_labels_blood_v001.nii.gz \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/render3d/combined \
  --formats stl ply obj \
  --target-max-faces 140000 \
  --report outputs/reports/combined_3d_render_stage001.md \
  --qa-report outputs/reports/combined_3d_render_mesh_qa_stage001.md
```

Generate a standard multi-view 3D render atlas with:

```bash
phantom-twin render-3d-atlas \
  --scene-spec outputs/render3d/combined/ct_org_case0_imagetbad_case125_3d_render_scene_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/render3d/combined_atlas \
  --report outputs/reports/combined_3d_render_atlas_stage001.md
```

Create a synthetic major-vessel network scaffold for flow modeling:

```bash
phantom-twin build-vascular-network-scaffold \
  --combined-spec outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/digital/vascular_network \
  --formats stl ply obj \
  --include-venous-return \
  --report outputs/reports/vascular_network_scaffold_stage001.md \
  --qa-report outputs/reports/vascular_network_scaffold_mesh_qa_stage001.md
```

Voxelize the vascular network scaffold into the combined digital phantom NIfTI:

```bash
phantom-twin voxelize-vascular-network \
  --graph outputs/digital/vascular_network/ct_org_case0_imagetbad_case125_vascular_network_graph_v001.yaml \
  --combined-labels outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_material_labels_blood_v001.nii.gz \
  --source-ct data/raw/ct_org/volumes/volume-0.nii.gz \
  --body-mask outputs/digital/torso/ct_org_case0_torso_body_mask_v001.nii.gz \
  --materials configs/materials.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/digital/vascular_network_voxelized \
  --contrast-mode arterial \
  --collision-cleanup nearest-centerline \
  --report outputs/reports/vascular_network_voxelized_stage001.md
```

Generate cleaned vascular-network 3D meshes, preview, and mesh QA:

```bash
phantom-twin render-vascular-network-3d \
  --context-labels outputs/digital/vascular_network_voxelized/ct_org_case0_imagetbad_case125_vascular_network_material_labels_contrast_v001.nii.gz \
  --arterial-mask outputs/digital/vascular_network_voxelized/ct_org_case0_imagetbad_case125_vascular_network_arterial_lumen_mask_v001.nii.gz \
  --venous-mask outputs/digital/vascular_network_voxelized/ct_org_case0_imagetbad_case125_vascular_network_venous_lumen_mask_v001.nii.gz \
  --flow-domain-labels outputs/digital/vascular_network_voxelized/ct_org_case0_imagetbad_case125_vascular_network_flow_domain_labels_v001.nii.gz \
  --vessel-wall-mask outputs/digital/vascular_network_voxelized/ct_org_case0_imagetbad_case125_vascular_network_vessel_wall_mask_v001.nii.gz \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/render3d/vascular_network_cleaned \
  --formats stl ply obj \
  --target-max-faces 140000 \
  --report outputs/reports/vascular_network_3d_render_stage001.md \
  --qa-report outputs/reports/vascular_network_3d_render_mesh_qa_stage001.md
```

Generate a multi-view atlas for the cleaned vascular-network 3D scene:

```bash
phantom-twin render-3d-atlas \
  --scene-spec outputs/render3d/vascular_network_cleaned/ct_org_case0_imagetbad_case125_vascular_network_3d_render_scene_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/render3d/vascular_network_cleaned_atlas \
  --report outputs/reports/vascular_network_3d_render_atlas_stage001.md
```

Build solver-facing flow boundary-condition metadata from the cleaned network:

```bash
phantom-twin build-flow-boundary-package \
  --voxelized-spec outputs/digital/vascular_network_voxelized/ct_org_case0_imagetbad_case125_vascular_network_voxelized_spec_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/sim/flow_boundary_conditions \
  --arterial-inlet-flow-ml-s 80.0 \
  --nominal-outlet-pressure-drop-pa 8000.0 \
  --venous-outlet-pressure-pa 667.0 \
  --boundary-slab-thickness-mm 5.0 \
  --report outputs/reports/flow_boundary_conditions_stage001.md
```

Build the first-pass steady 1D vascular flow model:

```bash
phantom-twin build-flow-1d-model \
  --graph outputs/digital/vascular_network/ct_org_case0_imagetbad_case125_vascular_network_graph_v001.yaml \
  --boundary-config outputs/sim/flow_boundary_conditions/ct_org_case0_imagetbad_case125_flow_boundary_conditions_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/sim/flow_1d \
  --blood-viscosity-cp 3.5 \
  --arterial-inlet-pressure-pa 13332.0 \
  --report outputs/reports/flow_1d_model_stage001.md
```

Upgrade the steady model into a first-pass pulsatile digital flow simulation:

```bash
phantom-twin build-pulsatile-flow-model \
  --flow-1d-model outputs/sim/flow_1d/ct_org_case0_imagetbad_case125_flow_1d_model_v001.yaml \
  --boundary-config outputs/sim/flow_boundary_conditions/ct_org_case0_imagetbad_case125_flow_boundary_conditions_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/sim/flow_pulsatile \
  --heart-rate-bpm 60.0 \
  --samples-per-cycle 160 \
  --settling-cycles 3 \
  --rcr-proximal-resistance-fraction 0.1 \
  --rcr-time-constant-s 1.2 \
  --venous-pulsatility-fraction 0.35 \
  --venous-phase-lag-fraction 0.15 \
  --report outputs/reports/flow_pulsatile_model_stage001.md
```

Build the coupled pulsatile graph-flow model with dynamic arterial outlet splits:

```bash
phantom-twin build-coupled-pulsatile-flow-model \
  --flow-1d-model outputs/sim/flow_1d/ct_org_case0_imagetbad_case125_flow_1d_model_v001.yaml \
  --boundary-config outputs/sim/flow_boundary_conditions/ct_org_case0_imagetbad_case125_flow_boundary_conditions_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/sim/flow_coupled_pulsatile \
  --heart-rate-bpm 60.0 \
  --samples-per-cycle 160 \
  --settling-cycles 3 \
  --rcr-proximal-resistance-fraction 0.1 \
  --rcr-time-constant-s 1.2 \
  --venous-pulsatility-fraction 0.35 \
  --venous-phase-lag-fraction 0.15 \
  --report outputs/reports/flow_coupled_pulsatile_model_stage001.md
```

Render the coupled vascular flow spatially inside the transparent digital phantom over time:

```bash
phantom-twin render-4d-flow \
  --graph outputs/digital/vascular_network/ct_org_case0_imagetbad_case125_vascular_network_graph_v001.yaml \
  --edge-timeseries outputs/sim/flow_coupled_pulsatile/ct_org_case0_imagetbad_case125_coupled_pulsatile_edge_timeseries_v001.csv \
  --node-timeseries outputs/sim/flow_coupled_pulsatile/ct_org_case0_imagetbad_case125_coupled_pulsatile_node_timeseries_v001.csv \
  --context-scene-spec outputs/render3d/vascular_network_cleaned/ct_org_case0_imagetbad_case125_vascular_network_3d_render_scene_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/sim/flow_4d_visualization \
  --color-by velocity \
  --frame-count 32 \
  --report outputs/reports/flow_4d_visualization_stage001.md
```

Build the first radiotherapy QA package with RT-ready material maps and DVH masks:

```bash
phantom-twin build-radiotherapy-qa-package \
  --combined-spec outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/radiotherapy/qa_package \
  --scenario blood \
  --target-radius-mm 12.0 \
  --ptv-margin-mm 5.0 \
  --report outputs/reports/radiotherapy_qa_package_stage001.md
```

Export the DICOM-RT-style planning handoff bundle and compare static versus
pulsatile vascular dose metrics:

```bash
phantom-twin build-rt-planning-bundle \
  --rt-package-spec outputs/radiotherapy/qa_package/ct_org_case0_imagetbad_case125_radiotherapy_qa_package_spec_v001.yaml \
  --coupled-flow-model outputs/sim/flow_coupled_pulsatile/ct_org_case0_imagetbad_case125_coupled_pulsatile_flow_model_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/radiotherapy/planning_bundle \
  --prescription-dose-gy 20.0 \
  --vascular-dose-sensitivity 0.015 \
  --export-dicom \
  --report outputs/reports/rt_planning_bundle_stage001.md
```

Run PyMedPhys gamma QA against the generated static and pulsatile dose states:

```bash
phantom-twin build-dose-gamma-qa \
  --pymedphys-eval-config outputs/radiotherapy/planning_bundle/ct_org_case0_imagetbad_case125_pymedphys_dose_eval_config_v001.yaml \
  --case-id ct_org_case0_imagetbad_case125 \
  --output-dir outputs/radiotherapy/dose_gamma_qa \
  --interp-fraction 3.0 \
  --max-gamma 2.0 \
  --random-subset 25000 \
  --global-gamma \
  --report outputs/reports/dose_gamma_qa_stage001.md
```
