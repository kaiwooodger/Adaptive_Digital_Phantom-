# Validation Protocol

Validation is the core of the project. The physical phantom is useful only if
it remains tied to measurable digital-twin targets.

## CT Density Validation

For each material coupon or organ module:

- Scan with the intended CT protocol.
- Register or identify the ROI.
- Record mean HU, standard deviation, min/max, and scan protocol.
- Compare against `configs/materials.yaml`.
- Update the material library only after measured evidence exists.

Initial MVP tolerances are intentionally forgiving. Tighten them after the first
coupon scan.

## Radiotherapy Validation

Minimum first workflow:

- Export/import phantom CT into the dose workflow.
- Compute or import a dose grid.
- Use PyMedPhys/pydicom to read dose data.
- Compare planned vs measured or simulated dose.
- Report gamma using a seed criterion such as 3 percent / 3 mm.

The first RT aim is repeatable dose comparison, not clinical-grade agreement.

## Vascular Flow Validation

Minimum first workflow:

- Record pump setting, fluid recipe, temperature, inlet pressure, outlet
  pressure, and measured flow.
- Test leaks at operating pressure.
- Compare measured pressure-flow curve to CFD or analytic expectations.
- Repeat at least three times and report repeatability.

Start steady-state. Add pulsatile flow after the geometry, ports, and sensors
are reliable.

## Reporting

Every validation report should include:

- Source data ID and version.
- Physical module version.
- Material recipe or coupon batch.
- Scan or bench-test protocol.
- Acceptance target.
- Pass/fail plus measured error.
