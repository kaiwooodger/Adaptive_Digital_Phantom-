# Data And Artifact Policy

This project intentionally separates source code from medical data and generated phantom artifacts.

## Do Not Commit

- Public dataset downloads unless the dataset license explicitly permits redistribution.
- Any anonymized or clinical CT/CTA/CTV/DICOM data without documented permission.
- Generated NIfTI volumes.
- Generated STL/PLY/OBJ/CAD files.
- Local Python, Docker, TotalSegmentator, or tool caches.
- Large release archives.

## Safe To Commit

- Source code.
- Tests.
- Configuration templates.
- Documentation.
- Small hand-written YAML/CSV examples that do not encode patient data.
- Small illustrative figures only when their source data license allows redistribution.

## Reproducibility Pattern

Use documentation and scripts to regenerate outputs locally:

```bash
phantom-twin build-product-case ...
phantom-twin render-vascular-network-3d ...
scripts/run_totalseg_docker_gi.py ...
```

Generated products should be published separately as release assets only after data-use and redistribution rights are checked.
