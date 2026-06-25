# Data Stage 001

Date: 2026-05-20

Goal: stage one CT-ORG anatomy case, one ImageTBAD vascular CTA case, inspect
the NIfTI files, and export first STL meshes for Phase 1 CAD work.

## CT-ORG Case 0

Downloaded from the Hugging Face mirror for fast single-case access:

- `https://huggingface.co/datasets/Angelou0516/ct-org/resolve/main/volumes/volume-0.nii.gz`
- `https://huggingface.co/datasets/Angelou0516/ct-org/resolve/main/labels/labels-0.nii.gz`

Local staged files:

- `data/raw/ct_org/volumes/volume-0.nii.gz`
- `data/raw/ct_org/labels/labels-0.nii.gz`

Inspection summary:

- Shape: `(512, 512, 75)`
- Spacing: `(0.703125, 0.703125, 5.0)` mm
- CT value range: `-3024` to `1410`
- Label values present: `0, 1, 2, 3, 4, 5`

Exported meshes:

- `outputs/meshes/ct_org_case0_lungs_v001.stl`
- `outputs/meshes/ct_org_case0_bone_v001.stl`
- `outputs/meshes/ct_org_case0_liver_v001.stl`
- `outputs/meshes/ct_org_case0_kidneys_v001.stl`
- `outputs/meshes/ct_org_case0_bladder_v001.stl`

## ImageTBAD Case 125

ImageTBAD is hosted on Kaggle as a 13.4 GB split archive. The current machine
has about 2 to 3 GB free, so the full archive could not be downloaded. Kaggle's
file-list API showed 19 split files. The final split component
`imageTBAD.change2zip` contains the central directory and complete entries for
case `125`, so case `125` was extracted without downloading the full archive.

Source endpoints used:

- Metadata: `https://www.kaggle.com/api/v1/datasets/list/xiaoweixumedicalai/imagetbad`
- Final split: `https://www.kaggle.com/api/v1/datasets/download/xiaoweixumedicalai/imagetbad/imageTBAD.change2zip`

Local staged files:

- `data/raw/imagetbad/case125/125_image.nii.gz`
- `data/raw/imagetbad/case125/125_label.nii.gz`

Inspection summary:

- Shape: `(512, 512, 174)`
- Spacing: `(1.0, 1.0, 1.0)` mm
- CTA value range: `-2000` to `4095`
- Label values present: `0, 1, 2, 3`
- Label voxel counts: true lumen `132986`, false lumen `122552`, thrombus `1091`

Exported meshes:

- `outputs/meshes/imagetbad_case125_true_lumen_v001.stl`
- `outputs/meshes/imagetbad_case125_false_lumen_v001.stl`
- `outputs/meshes/imagetbad_case125_false_lumen_thrombus_v001.stl`
- `outputs/meshes/imagetbad_case125_aorta_dissection_combined_v001.stl`

## Implementation Notes

The mesh exporter now rounds scaled floating-point label values before matching
integer label IDs. This was needed because CT-ORG stores labels as values such
as `2.9999999993` rather than exact integer `3`.

The mesh exporter also supports multiple label IDs in one command, which was
used to create the combined ImageTBAD aorta/dissection mesh:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m phantom_twin.cli export-label-mesh \
  --labels data/raw/imagetbad/case125/125_label.nii.gz \
  --label-id 1 2 3 \
  --output outputs/meshes/imagetbad_case125_aorta_dissection_combined_v001.stl
```

## Next Step

Open the STL files in 3D Slicer, Blender, MeshLab, or FreeCAD to visually check
orientation and surface quality. After that, create smoothed/decimated CAD-ready
versions and add vessel inlet/outlet ports for the first flow-loop module.
