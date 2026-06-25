# CAD Workflow

The CAD model should be generated from the digital twin, not drawn from scratch.
Manual CAD edits are allowed, but they must be traceable.

## Recommended Flow

```text
NIfTI/DICOM volume
  -> label masks
  -> marching-cubes surface mesh
  -> smoothing/repair
  -> CAD module design
  -> manufacturing export
  -> scanned physical phantom
  -> registration back to digital twin
```

## Tool Roles

- 3D Slicer: visual inspection, segmentation review, STL/OBJ export.
- Python: repeatable NIfTI inspection, label extraction, mesh export.
- MeshLab/Blender: mesh cleanup, decimation, smoothing, surface checks.
- FreeCAD/OpenCascade: mechanical design, ports, inserts, fixtures, STEP export.
- Gmsh/SimVascular: vascular/CFD mesh preparation.

## CAD Rules

- Preserve a one-to-one mapping from mesh file to source case, label, and
  material ID.
- Add alignment features early: dowel holes, keyed modules, fiducials, and
  scanner setup marks.
- Do not route fluid channels through dosimeter positions unless intentionally
  testing that interaction.
- Avoid single-use monolithic prints. Make organs and QA inserts replaceable.

## Mesh Naming

Use predictable names:

```text
{dataset}_{case}_{structure}_{version}.{ext}
ct_org_case000_lungs_v001.stl
imagetbad_case014_true_lumen_v001.stl
```
