from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def edge_radius_at_fraction(edge: dict[str, Any], fraction: float, radius_scale: float = 1.0, minimum_radius_mm: float = 0.05) -> float:
    """Return edge radius at normalized station, honoring optional radius_profile."""

    station = min(max(float(fraction), 0.0), 1.0)
    radius_start = _safe_float(edge.get("radius_start_mm", edge.get("radius_mm")), 1.0)
    radius_end = _safe_float(edge.get("radius_end_mm"), radius_start)
    fallback = radius_start + (radius_end - radius_start) * station

    profile = edge.get("radius_profile")
    if not isinstance(profile, list) or not profile:
        return max(float(fallback) * float(radius_scale), float(minimum_radius_mm))

    parsed: list[tuple[float, float]] = []
    for item in profile:
        if not isinstance(item, dict):
            continue
        item_station = _safe_float(item.get("station", item.get("fraction")), -1.0)
        item_radius = _safe_float(item.get("radius_mm"), float("nan"))
        if 0.0 <= item_station <= 1.0 and item_radius == item_radius:
            parsed.append((item_station, item_radius))

    if not parsed:
        return max(float(fallback) * float(radius_scale), float(minimum_radius_mm))

    parsed = sorted(parsed, key=lambda item: item[0])
    if station <= parsed[0][0]:
        radius = parsed[0][1]
    elif station >= parsed[-1][0]:
        radius = parsed[-1][1]
    else:
        radius = fallback
        for left, right in zip(parsed[:-1], parsed[1:]):
            left_station, left_radius = left
            right_station, right_radius = right
            if left_station <= station <= right_station:
                width = max(right_station - left_station, 1e-9)
                local_t = (station - left_station) / width
                radius = left_radius + (right_radius - left_radius) * local_t
                break

    return max(float(radius) * float(radius_scale), float(minimum_radius_mm))


def edge_radius_profile_max(edge: dict[str, Any]) -> float:
    radii = [
        _safe_float(edge.get("radius_start_mm", edge.get("radius_mm")), 0.0),
        _safe_float(edge.get("radius_end_mm", edge.get("radius_mm")), 0.0),
    ]
    profile = edge.get("radius_profile")
    if isinstance(profile, list):
        for item in profile:
            if isinstance(item, dict):
                radii.append(_safe_float(item.get("radius_mm"), 0.0))
    return max(radii) if radii else 0.0
