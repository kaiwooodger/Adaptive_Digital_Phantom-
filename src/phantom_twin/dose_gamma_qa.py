from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import warnings
from typing import Any

import numpy as np


def _import_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import nibabel as nib  # type: ignore
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Dose gamma QA requires matplotlib, nibabel, and PyYAML.") from exc
    return plt, nib, yaml


def _import_pymedphys():
    try:
        import pymedphys  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Dose gamma QA requires PyMedPhys for gamma analysis. "
            "Install with: /opt/anaconda3/bin/python -m pip install pymedphys"
        ) from exc
    return pymedphys


@dataclass(frozen=True)
class GammaStateResult:
    state: str
    evaluated_dose_path: str
    gamma_map_path: str | None
    dose_difference_path: str | None
    finite_gamma_points: int
    passing_gamma_points: int
    pass_rate_percent: float
    mean_gamma: float
    median_gamma: float
    p95_gamma: float
    max_gamma_value: float
    mean_abs_dose_difference_gy: float
    max_abs_dose_difference_gy: float
    mean_dose_difference_percent: float
    max_abs_dose_difference_percent: float


@dataclass(frozen=True)
class DoseGammaQAResult:
    case_id: str
    output_dir: str
    reference_dose_path: str
    summary_csv_path: str
    spec_yaml_path: str
    preview_png_path: str
    report_path: str
    dose_percent_threshold: float
    distance_mm_threshold: float
    lower_percent_dose_cutoff: float
    interp_fraction: float
    max_gamma: float | None
    local_gamma: bool
    random_subset: int | None
    random_seed: int
    global_normalisation_gy: float
    lower_dose_cutoff_gy: float
    volume_outputs_written: bool
    pymedphys_version: str
    state_results: tuple[GammaStateResult, ...]
    notes: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    *_, yaml = _import_dependencies()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file is not a mapping: {path}")
    return data


def _resolve_path(raw_path: str | Path, reference_yaml_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = reference_yaml_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _write_nifti(path: Path, data: np.ndarray, reference_image) -> None:
    _, nib, _ = _import_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), reference_image.affine, reference_image.header)
    image.set_data_dtype(np.float32)
    nib.save(image, str(path))


def _axes_for_image(image, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    spacing = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
    return tuple(np.arange(shape[axis], dtype=np.float64) * spacing[axis] for axis in range(len(shape)))


def _finite_stat(values: np.ndarray, fn, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(fn(finite))


def _run_pymedphys_gamma(
    axes: tuple[np.ndarray, ...],
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    dose_percent_threshold: float,
    distance_mm_threshold: float,
    lower_percent_dose_cutoff: float,
    interp_fraction: float,
    max_gamma: float | None,
    local_gamma: bool,
    random_subset: int | None,
    global_normalisation: float,
    random_seed: int,
) -> np.ndarray:
    pymedphys = _import_pymedphys()
    if random_subset is not None:
        np.random.seed(random_seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        gamma = pymedphys.gamma(
            axes,
            reference_dose,
            axes,
            evaluated_dose,
            dose_percent_threshold=dose_percent_threshold,
            distance_mm_threshold=distance_mm_threshold,
            lower_percent_dose_cutoff=lower_percent_dose_cutoff,
            interp_fraction=interp_fraction,
            max_gamma=max_gamma,
            local_gamma=local_gamma,
            global_normalisation=global_normalisation,
            skip_once_passed=True,
            random_subset=random_subset,
            ram_available=2**31,
            quiet=True,
        )
    return gamma.astype(np.float32)


def _summarise_state(
    state: str,
    evaluated_path: str,
    gamma_map_path: str | None,
    dose_difference_path: str | None,
    gamma: np.ndarray,
    dose_difference: np.ndarray,
    comparison_mask: np.ndarray,
    global_normalisation_gy: float,
) -> GammaStateResult:
    finite = np.isfinite(gamma)
    finite_values = gamma[finite]
    passing = finite_values <= 1.0
    diff_values = dose_difference[comparison_mask]
    percent_diff = 100.0 * diff_values / max(global_normalisation_gy, 1e-9)
    finite_count = int(finite_values.size)
    passing_count = int(passing.sum())
    pass_rate = 0.0 if finite_count == 0 else 100.0 * passing_count / finite_count
    return GammaStateResult(
        state=state,
        evaluated_dose_path=evaluated_path,
        gamma_map_path=gamma_map_path,
        dose_difference_path=dose_difference_path,
        finite_gamma_points=finite_count,
        passing_gamma_points=passing_count,
        pass_rate_percent=float(pass_rate),
        mean_gamma=_finite_stat(finite_values, np.mean),
        median_gamma=_finite_stat(finite_values, np.median),
        p95_gamma=_finite_stat(finite_values, lambda values: np.percentile(values, 95.0)),
        max_gamma_value=_finite_stat(finite_values, np.max),
        mean_abs_dose_difference_gy=_finite_stat(np.abs(diff_values), np.mean),
        max_abs_dose_difference_gy=_finite_stat(np.abs(diff_values), np.max),
        mean_dose_difference_percent=_finite_stat(percent_diff, np.mean),
        max_abs_dose_difference_percent=_finite_stat(np.abs(percent_diff), np.max),
    )


def _write_summary_csv(path: Path, state_results: tuple[GammaStateResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "state",
        "finite_gamma_points",
        "passing_gamma_points",
        "pass_rate_percent",
        "mean_gamma",
        "median_gamma",
        "p95_gamma",
        "max_gamma_value",
        "mean_abs_dose_difference_gy",
        "max_abs_dose_difference_gy",
        "mean_dose_difference_percent",
        "max_abs_dose_difference_percent",
        "evaluated_dose_path",
        "gamma_map_path",
        "dose_difference_path",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for result in state_results:
            writer.writerow(
                {
                    "state": result.state,
                    "finite_gamma_points": result.finite_gamma_points,
                    "passing_gamma_points": result.passing_gamma_points,
                    "pass_rate_percent": f"{result.pass_rate_percent:.6f}",
                    "mean_gamma": f"{result.mean_gamma:.6f}",
                    "median_gamma": f"{result.median_gamma:.6f}",
                    "p95_gamma": f"{result.p95_gamma:.6f}",
                    "max_gamma_value": f"{result.max_gamma_value:.6f}",
                    "mean_abs_dose_difference_gy": f"{result.mean_abs_dose_difference_gy:.6f}",
                    "max_abs_dose_difference_gy": f"{result.max_abs_dose_difference_gy:.6f}",
                    "mean_dose_difference_percent": f"{result.mean_dose_difference_percent:.6f}",
                    "max_abs_dose_difference_percent": f"{result.max_abs_dose_difference_percent:.6f}",
                    "evaluated_dose_path": result.evaluated_dose_path,
                    "gamma_map_path": result.gamma_map_path or "",
                    "dose_difference_path": result.dose_difference_path or "",
                }
            )


def _write_spec(path: Path, config_path: Path, result: DoseGammaQAResult) -> None:
    *_, yaml = _import_dependencies()
    payload = {
        "case_id": result.case_id,
        "qa_type": "pymedphys_gamma_static_vs_pulsatile",
        "inputs": {
            "pymedphys_eval_config": str(config_path),
            "reference_dose": result.reference_dose_path,
        },
        "gamma_settings": {
            "dose_percent_threshold": result.dose_percent_threshold,
            "distance_mm_threshold": result.distance_mm_threshold,
            "lower_percent_dose_cutoff": result.lower_percent_dose_cutoff,
            "interp_fraction": result.interp_fraction,
            "max_gamma": result.max_gamma,
            "local_gamma": result.local_gamma,
            "random_subset": result.random_subset,
            "random_seed": result.random_seed,
            "global_normalisation_gy": result.global_normalisation_gy,
            "lower_dose_cutoff_gy": result.lower_dose_cutoff_gy,
        },
        "outputs": {
            "summary_csv": result.summary_csv_path,
            "preview_png": result.preview_png_path,
            "report": result.report_path,
            "volume_outputs_written": result.volume_outputs_written,
        },
        "state_results": [
            {
                "state": state.state,
                "pass_rate_percent": state.pass_rate_percent,
                "finite_gamma_points": state.finite_gamma_points,
                "p95_gamma": state.p95_gamma,
                "max_gamma_value": state.max_gamma_value,
                "gamma_map": state.gamma_map_path,
                "dose_difference": state.dose_difference_path,
            }
            for state in result.state_results
        ],
        "pymedphys_version": result.pymedphys_version,
        "notes": list(result.notes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_preview(
    path: Path,
    reference_dose: np.ndarray,
    state_results: tuple[GammaStateResult, ...],
    gamma_maps: dict[str, np.ndarray],
    dose_differences: dict[str, np.ndarray],
) -> None:
    plt, _, _ = _import_dependencies()
    z = int(np.unravel_index(np.nanargmax(reference_dose), reference_dose.shape)[2])
    rows = max(len(state_results), 1)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.0 * rows), dpi=150)
    if rows == 1:
        axes = np.asarray([axes])
    for row, state in enumerate(state_results):
        diff = dose_differences[state.state]
        gamma = gamma_maps[state.state]
        finite_gamma = gamma[np.isfinite(gamma)]
        ax0, ax1, ax2 = axes[row]
        im0 = ax0.imshow(diff[:, :, z].T, cmap="coolwarm", origin="lower", vmin=-0.08, vmax=0.08)
        ax0.set_title(f"{state.state}: dose diff Gy")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        gamma_slice = gamma[:, :, z].T
        im1 = ax1.imshow(gamma_slice, cmap="viridis", origin="lower", vmin=0.0, vmax=1.2)
        ax1.set_title(f"{state.state}: sampled gamma")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        if finite_gamma.size:
            ax2.hist(finite_gamma, bins=40, range=(0.0, max(1.2, float(np.nanmax(finite_gamma)))), color="#1d4ed8")
            ax2.axvline(1.0, color="#b91c1c", linewidth=1.5, label="gamma=1")
            ax2.legend(fontsize=8)
        ax2.set_title(f"pass {state.pass_rate_percent:.2f}% | p95 {state.p95_gamma:.3f}")
        ax2.set_xlabel("gamma")
        ax2.set_ylabel("sampled voxels")
        for ax in (ax0, ax1):
            ax.set_xlabel("i")
            ax.set_ylabel("j")
    fig.suptitle(f"PyMedPhys gamma QA at max-dose slice z={z}", y=1.0, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _format_report(result: DoseGammaQAResult) -> str:
    lines = [
        "# Dose Gamma QA Stage 001",
        "",
        f"Case ID: `{result.case_id}`",
        f"PyMedPhys version: `{result.pymedphys_version}`",
        "",
        "## Gamma Settings",
        "",
        f"- Criteria: {result.dose_percent_threshold:.1f}% / {result.distance_mm_threshold:.1f} mm",
        f"- Lower dose cutoff: {result.lower_percent_dose_cutoff:.1f}% = {result.lower_dose_cutoff_gy:.3f} Gy",
        f"- Local gamma: {result.local_gamma}",
        f"- Interpolation fraction: {result.interp_fraction:.2f}",
        f"- Max gamma search: {result.max_gamma}",
        f"- Random subset: {result.random_subset}",
        f"- Random seed: {result.random_seed}",
        "",
        "## Results",
        "",
        "| state | pass rate % | sampled points | mean gamma | p95 gamma | max gamma | max abs diff Gy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for state in result.state_results:
        lines.append(
            f"| `{state.state}` | {state.pass_rate_percent:.3f} | {state.finite_gamma_points} | "
            f"{state.mean_gamma:.4f} | {state.p95_gamma:.4f} | {state.max_gamma_value:.4f} | "
            f"{state.max_abs_dose_difference_gy:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary CSV: `{Path(result.summary_csv_path).name}`",
            f"- QA spec: `{Path(result.spec_yaml_path).name}`",
            f"- Preview PNG: `{Path(result.preview_png_path).name}`",
            "",
            "## Interpretation",
            "",
            "- Gamma was calculated with PyMedPhys using the static dose as reference and pulsatile states as evaluated dose grids.",
            "- The current run uses a reproducible random subset to keep 3D gamma practical on the full 512 x 512 x 75 dose grid.",
            "- Dose-difference and sampled gamma NIfTI maps are written only when volume outputs are enabled.",
            "- These doses are still synthetic engineering test patterns, so this is a pipeline validation rather than clinical dose QA.",
            "",
            "## Notes",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def build_dose_gamma_qa(
    pymedphys_eval_config_path: str | Path,
    output_dir: str | Path = "outputs/radiotherapy/dose_gamma_qa",
    case_id: str = "ct_org_case0_imagetbad_case125",
    dose_percent_threshold: float | None = None,
    distance_mm_threshold: float | None = None,
    lower_percent_dose_cutoff: float | None = None,
    interp_fraction: float = 3.0,
    max_gamma: float | None = 2.0,
    local_gamma: bool | None = None,
    random_subset: int | None = 25000,
    random_seed: int = 20260526,
    write_volume_outputs: bool = True,
    report_path: str | Path | None = "outputs/reports/dose_gamma_qa_stage001.md",
) -> DoseGammaQAResult:
    plt, nib, _ = _import_dependencies()
    del plt
    pymedphys = _import_pymedphys()
    config_path = Path(pymedphys_eval_config_path)
    config = _load_yaml(config_path)
    gamma_defaults = config.get("gamma_defaults", {})
    if not isinstance(gamma_defaults, dict):
        gamma_defaults = {}
    dose_percent = float(
        dose_percent_threshold
        if dose_percent_threshold is not None
        else gamma_defaults.get("dose_percent_threshold", 3.0)
    )
    distance_mm = float(
        distance_mm_threshold
        if distance_mm_threshold is not None
        else gamma_defaults.get("distance_mm_threshold", 3.0)
    )
    lower_cutoff_percent = float(
        lower_percent_dose_cutoff
        if lower_percent_dose_cutoff is not None
        else gamma_defaults.get("lower_percent_dose_cutoff", 10.0)
    )
    local = bool(local_gamma if local_gamma is not None else gamma_defaults.get("local_gamma", False))

    reference_payload = config.get("reference_dose", {})
    evaluated_payloads = config.get("evaluated_doses", [])
    if not isinstance(reference_payload, dict) or "nifti" not in reference_payload:
        raise ValueError("PyMedPhys config is missing reference_dose.nifti")
    if not isinstance(evaluated_payloads, list) or not evaluated_payloads:
        raise ValueError("PyMedPhys config is missing evaluated_doses")

    reference_path = _resolve_path(str(reference_payload["nifti"]), config_path)
    reference_image = nib.load(str(reference_path))
    reference_dose = np.asanyarray(reference_image.dataobj).astype(np.float32)
    axes = _axes_for_image(reference_image, reference_dose.shape)
    global_normalisation = float(np.nanmax(reference_dose))
    lower_cutoff_gy = global_normalisation * lower_cutoff_percent / 100.0
    comparison_mask = np.isfinite(reference_dose) & (reference_dose >= lower_cutoff_gy)

    output = Path(output_dir)
    gamma_dir = output / "gamma_maps"
    diff_dir = output / "dose_difference"
    state_results: list[GammaStateResult] = []
    gamma_maps: dict[str, np.ndarray] = {}
    dose_differences: dict[str, np.ndarray] = {}
    for index, payload in enumerate(evaluated_payloads):
        if not isinstance(payload, dict) or "nifti" not in payload:
            continue
        state = str(payload.get("state", f"evaluated_{index + 1}"))
        evaluated_path = _resolve_path(str(payload["nifti"]), config_path)
        evaluated_image = nib.load(str(evaluated_path))
        evaluated_dose = np.asanyarray(evaluated_image.dataobj).astype(np.float32)
        if evaluated_dose.shape != reference_dose.shape:
            raise ValueError(f"Evaluated dose {state} shape does not match reference dose")
        gamma = _run_pymedphys_gamma(
            axes=axes,
            reference_dose=reference_dose,
            evaluated_dose=evaluated_dose,
            dose_percent_threshold=dose_percent,
            distance_mm_threshold=distance_mm,
            lower_percent_dose_cutoff=lower_cutoff_percent,
            interp_fraction=interp_fraction,
            max_gamma=max_gamma,
            local_gamma=local,
            random_subset=random_subset,
            global_normalisation=global_normalisation,
            random_seed=random_seed + index,
        )
        dose_difference = (evaluated_dose - reference_dose).astype(np.float32)
        gamma_path = gamma_dir / f"{case_id}_{state}_gamma_{dose_percent:.0f}pct_{distance_mm:.0f}mm_v001.nii.gz"
        difference_path = diff_dir / f"{case_id}_{state}_minus_static_dose_difference_v001.nii.gz"
        gamma_path_out: str | None = str(gamma_path)
        difference_path_out: str | None = str(difference_path)
        if write_volume_outputs:
            _write_nifti(gamma_path, gamma, reference_image)
            _write_nifti(difference_path, dose_difference, reference_image)
        else:
            gamma_path_out = None
            difference_path_out = None
        state_result = _summarise_state(
            state=state,
            evaluated_path=str(evaluated_path),
            gamma_map_path=gamma_path_out,
            dose_difference_path=difference_path_out,
            gamma=gamma,
            dose_difference=dose_difference,
            comparison_mask=comparison_mask,
            global_normalisation_gy=global_normalisation,
        )
        state_results.append(state_result)
        gamma_maps[state] = gamma
        dose_differences[state] = dose_difference

    summary_csv = output / f"{case_id}_dose_gamma_qa_summary_v001.csv"
    spec_yaml = output / f"{case_id}_dose_gamma_qa_spec_v001.yaml"
    preview_png = output / f"{case_id}_dose_gamma_qa_preview_v001.png"
    report = Path(report_path) if report_path else output / f"{case_id}_dose_gamma_qa_report_v001.md"
    results_tuple = tuple(state_results)
    notes = [
        "gamma_calculated_with_pymedphys",
        "gamma_maps_are_random_subset_sampled_when_random_subset_is_set",
        "synthetic_dose_inputs_are_engineering_test_patterns_not_clinical_calculations",
    ]
    if write_volume_outputs:
        notes.append("dose_difference_maps_are_full_volume")
    else:
        notes.append("volume_outputs_skipped_for_disk_light_run")
    result = DoseGammaQAResult(
        case_id=case_id,
        output_dir=str(output),
        reference_dose_path=str(reference_path),
        summary_csv_path=str(summary_csv),
        spec_yaml_path=str(spec_yaml),
        preview_png_path=str(preview_png),
        report_path=str(report),
        dose_percent_threshold=dose_percent,
        distance_mm_threshold=distance_mm,
        lower_percent_dose_cutoff=lower_cutoff_percent,
        interp_fraction=float(interp_fraction),
        max_gamma=max_gamma,
        local_gamma=local,
        random_subset=random_subset,
        random_seed=random_seed,
        global_normalisation_gy=global_normalisation,
        lower_dose_cutoff_gy=lower_cutoff_gy,
        volume_outputs_written=write_volume_outputs,
        pymedphys_version=str(getattr(pymedphys, "__version__", "unknown")),
        state_results=results_tuple,
        notes=tuple(notes),
    )
    _write_summary_csv(summary_csv, results_tuple)
    _write_preview(preview_png, reference_dose, results_tuple, gamma_maps, dose_differences)
    _write_spec(spec_yaml, config_path, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_format_report(result))
    return result


def format_dose_gamma_qa_result(result: DoseGammaQAResult) -> str:
    lines = [
        "Dose gamma QA completed",
        f"Case ID: {result.case_id}",
        f"PyMedPhys version: {result.pymedphys_version}",
        f"Criteria: {result.dose_percent_threshold:.1f}% / {result.distance_mm_threshold:.1f} mm",
        f"Lower cutoff: {result.lower_percent_dose_cutoff:.1f}% ({result.lower_dose_cutoff_gy:.3f} Gy)",
        f"Random subset: {result.random_subset}",
        f"Summary CSV: {result.summary_csv_path}",
        f"Preview PNG: {result.preview_png_path}",
        f"Spec YAML: {result.spec_yaml_path}",
        f"Report: {result.report_path}",
    ]
    for state in result.state_results:
        lines.append(
            f"{state.state}: pass={state.pass_rate_percent:.3f}% "
            f"points={state.finite_gamma_points} p95={state.p95_gamma:.4f} "
            f"max={state.max_gamma_value:.4f}"
        )
    return "\n".join(lines)
