# Phase 1 Build Plan

The MVP is a modular thorax/upper-abdomen phantom with three linked validation
tracks: CT density, radiotherapy, and vascular flow.

## Build Principle

Do not start by printing a full body. Start with a digital twin that can survive
measurement:

1. Choose one reference CT case and one vascular CTA case.
2. Export coarse organ/vessel meshes.
3. Build material coupons.
4. CT scan the coupons.
5. Update the material library from measured values.
6. Only then design the first torso module.

## Work Packages

### WP1: Data and Labels

- Download one CT-ORG case.
- Confirm image spacing, affine, HU range, and label IDs.
- Export lung, liver, kidney, bladder, and bone STL files.
- Choose one vascular source and inspect its labels.

### WP2: Material Coupon Library

- Build small coupons for lung, soft tissue, liver, blood-equivalent fluid,
  trabecular bone, cortical bone, and vessel wall.
- Scan coupons using the intended CT protocol.
- Record mean HU, standard deviation, density, and manufacturing notes.

### WP3: CAD Alpha

- Create a torso shell from the external contour or an approximate body volume.
- Add removable lung, liver, and bone modules.
- Add a vessel-routing module with inlet/outlet ports.
- Add fiducials and sensor/dosimeter pockets.

### WP4: Radiotherapy Validation

- Import CT scan of the phantom into a research TPS or Monte Carlo workflow.
- Use PyMedPhys/pydicom for dose-grid extraction and comparison.
- Use a simple gamma report as the first objective RT metric.

### WP5: Flow Validation

- Convert vascular geometry to a smooth lumen mesh.
- Define inlet/outlet ports and measurable sections.
- Run a bench flow loop with pressure and flow sensors.
- Keep CFD initially simple: steady incompressible flow before pulsatile flow.

## First Acceptance Target

The first win is not realism everywhere. It is traceability:

- The dataset source is documented.
- The mesh comes from a known label volume.
- The material target is documented.
- The manufactured coupon is scanned.
- The measured HU/RED is fed back into the digital twin.
