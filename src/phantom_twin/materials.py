from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MaterialTarget:
    id: str
    label: str
    category: str
    target_hu: tuple[float, float]
    target_mass_density_g_cm3: float
    target_relative_electron_density: float
    candidate_materials: tuple[str, ...]
    validation_method: str

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "MaterialTarget":
        required = [
            "id",
            "label",
            "category",
            "target_hu",
            "target_mass_density_g_cm3",
            "target_relative_electron_density",
            "candidate_materials",
            "validation_method",
        ]
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Material is missing required keys: {missing}")

        target_hu = item["target_hu"]
        if not isinstance(target_hu, list | tuple) or len(target_hu) != 2:
            raise ValueError(f"Material {item['id']} target_hu must be [min, max]")

        hu_min, hu_max = float(target_hu[0]), float(target_hu[1])
        if hu_min > hu_max:
            raise ValueError(f"Material {item['id']} target_hu min is greater than max")

        density = float(item["target_mass_density_g_cm3"])
        red = float(item["target_relative_electron_density"])
        if density <= 0:
            raise ValueError(f"Material {item['id']} density must be positive")
        if red <= 0:
            raise ValueError(f"Material {item['id']} RED must be positive")

        candidates = tuple(str(value) for value in item["candidate_materials"])
        if not candidates:
            raise ValueError(f"Material {item['id']} needs at least one candidate material")

        return cls(
            id=str(item["id"]),
            label=str(item["label"]),
            category=str(item["category"]),
            target_hu=(hu_min, hu_max),
            target_mass_density_g_cm3=density,
            target_relative_electron_density=red,
            candidate_materials=candidates,
            validation_method=str(item["validation_method"]),
        )


@dataclass(frozen=True)
class MaterialLibrary:
    materials: tuple[MaterialTarget, ...]
    metadata: dict[str, Any]

    @property
    def by_id(self) -> dict[str, MaterialTarget]:
        return {material.id: material for material in self.materials}

    @property
    def categories(self) -> dict[str, list[MaterialTarget]]:
        result: dict[str, list[MaterialTarget]] = {}
        for material in self.materials:
            result.setdefault(material.category, []).append(material)
        return result


def load_material_library(path: str | Path) -> MaterialLibrary:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    raw_materials = data.get("materials")
    if not isinstance(raw_materials, list):
        raise ValueError(f"{config_path} must contain a materials list")

    materials = tuple(MaterialTarget.from_mapping(item) for item in raw_materials)
    ids = [material.id for material in materials]
    duplicates = sorted({material_id for material_id in ids if ids.count(material_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate material IDs: {duplicates}")

    return MaterialLibrary(materials=materials, metadata=data.get("metadata", {}))


def summarize_materials(library: MaterialLibrary) -> str:
    lines = [
        f"Material targets: {len(library.materials)}",
        f"Categories: {', '.join(sorted(library.categories))}",
        "",
        "| id | category | HU target | density | RED |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for material in sorted(library.materials, key=lambda item: (item.category, item.id)):
        hu_min, hu_max = material.target_hu
        lines.append(
            "| "
            f"{material.id} | {material.category} | {hu_min:g} to {hu_max:g} | "
            f"{material.target_mass_density_g_cm3:g} | "
            f"{material.target_relative_electron_density:g} |"
        )
    return "\n".join(lines)
