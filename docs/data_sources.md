# Data Sources

Phase 1 needs anatomy, density information, and vascular geometry. No single
public source gives all of that cleanly, so the project starts with a practical
multi-source strategy.

## Selected Starting Sources

**CT-ORG** is the immediate CT anatomy source. It includes 140 CT scans with
NIfTI organ segmentations for liver, bladder, lungs, kidneys, bone, and partial
brain. TCIA lists the collection as 16.9 GB with a CC BY 3.0 license and DOI
`10.7937/tcia.2019.tt7f4v7o`.

Use it first because it is simple enough to prove the pipeline:

- Load CT and label volumes.
- Export lungs, liver, kidney, bladder, and bone meshes.
- Map labels to material targets.
- Start CT density validation reporting.

**Healthy-Total-Body-CTs** is the best whole-body upgrade target. TCIA describes
it as low-dose whole-body CT images and tissue segmentations for 30 healthy
adult participants, with 37 tissue segmentations. The CT images are under NIH
controlled data access, while the segmentations are listed as CC BY 4.0.

Use it once access is sorted because it better matches the long-term goal:

- Healthy whole-body proportions.
- More tissue categories.
- Whole-body segmentation package.

**ImageTBAD** is the first vascular geometry source. It is a public CTA dataset
focused on type-B aortic dissection with true lumen, false lumen, and thrombus
annotations. It is pathology-specific, so the first use is engineering geometry:
a pumpable aorta/large-vessel module, not a healthy reference anatomy.

## Upgrade Sources

**XCAT** is a strong computational-phantom target for 4D cardiac/respiratory
motion and whole-body digital twin work, but it is licensed rather than
open-source.

**ICRP reference phantoms** are useful for organ mass, tissue composition, and
dose-reference assumptions. Treat them as reference data with licensing and
redistribution constraints, not as the first CAD source.

## Immediate Decision

Use CT-ORG plus ImageTBAD first. This keeps us moving without waiting on
controlled-access approvals, while leaving a clear path to Healthy-Total-Body,
XCAT, or ICRP models later.
