from __future__ import annotations

from pathlib import Path

import yaml

from .datasets import DatasetManifest
from .materials import MaterialLibrary


def load_phase1_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def build_phase1_summary(
    phase1_config: dict,
    material_library: MaterialLibrary,
    dataset_manifest: DatasetManifest,
) -> str:
    project = phase1_config.get("project", {})
    anatomical_region = phase1_config.get("anatomical_region", {})
    milestones = phase1_config.get("milestones", [])

    lines = [
        f"# {project.get('build_name', 'Phase 1 Phantom MVP')}",
        "",
        f"Objective: {project.get('objective', '').strip()}",
        "",
        f"Initial region: {anatomical_region.get('initial_region', 'unknown')}",
        f"Material targets: {len(material_library.materials)}",
        f"Dataset sources: {len(dataset_manifest.sources)}",
        "",
        "## Phase 1 Data Choices",
    ]

    for role, source_id in dataset_manifest.selected_for_phase1.items():
        source = dataset_manifest.by_id[source_id]
        lines.append(f"- {role}: {source.name} ({source.id})")

    lines.extend(["", "## Milestones"])
    for milestone in milestones:
        lines.append(
            f"- {milestone.get('id')}: {milestone.get('name')} - "
            f"{milestone.get('exit_criteria')}"
        )

    lines.extend(
        [
            "",
            "## Immediate Next Build Step",
            "Download or stage one CT-ORG case, inspect it, and export the first lung/bone meshes.",
        ]
    )
    return "\n".join(lines)
