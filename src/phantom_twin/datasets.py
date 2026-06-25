from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSource:
    id: str
    name: str
    role: str
    modality: tuple[str, ...]
    anatomy: tuple[str, ...]
    license: str
    access: str
    url: str
    citation_required: bool
    limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "DatasetSource":
        required = [
            "id",
            "name",
            "role",
            "modality",
            "anatomy",
            "license",
            "access",
            "url",
            "citation_required",
        ]
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Dataset source is missing required keys: {missing}")

        return cls(
            id=str(item["id"]),
            name=str(item["name"]),
            role=str(item["role"]),
            modality=tuple(str(value) for value in item["modality"]),
            anatomy=tuple(str(value) for value in item["anatomy"]),
            license=str(item["license"]),
            access=str(item["access"]),
            url=str(item["url"]),
            citation_required=bool(item["citation_required"]),
            limitations=tuple(str(value) for value in item.get("limitations", [])),
        )


@dataclass(frozen=True)
class DatasetManifest:
    selected_for_phase1: dict[str, str]
    sources: tuple[DatasetSource, ...]
    metadata: dict[str, Any]

    @property
    def by_id(self) -> dict[str, DatasetSource]:
        return {source.id: source for source in self.sources}


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError(f"{config_path} must contain a sources list")

    sources = tuple(DatasetSource.from_mapping(item) for item in raw_sources)
    ids = [source.id for source in sources]
    duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate dataset IDs: {duplicates}")

    selected = data.get("selected_for_phase1", {})
    missing_selected = sorted(set(selected.values()) - set(ids))
    if missing_selected:
        raise ValueError(f"selected_for_phase1 references unknown datasets: {missing_selected}")

    return DatasetManifest(
        selected_for_phase1=dict(selected),
        sources=sources,
        metadata=data.get("metadata", {}),
    )


def summarize_datasets(manifest: DatasetManifest) -> str:
    lines = [
        f"Dataset sources: {len(manifest.sources)}",
        "",
        "Phase 1 selections:",
    ]
    for role, source_id in manifest.selected_for_phase1.items():
        lines.append(f"- {role}: {source_id}")

    lines.extend(
        [
            "",
            "| id | role | modality | access/license |",
            "| --- | --- | --- | --- |",
        ]
    )
    for source in sorted(manifest.sources, key=lambda item: item.id):
        modality = ", ".join(source.modality)
        lines.append(f"| {source.id} | {source.role} | {modality} | {source.access}; {source.license} |")
    return "\n".join(lines)
