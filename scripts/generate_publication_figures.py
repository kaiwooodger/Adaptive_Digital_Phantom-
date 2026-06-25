from __future__ import annotations

import csv
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "publication"


COLORS = {
    "ink": "#17212b",
    "muted": "#5d6975",
    "grid": "#d8dee6",
    "paper": "#ffffff",
    "panel": "#f7f4ee",
    "blue": "#2878b8",
    "cyan": "#2bb3c0",
    "red": "#dc3b2a",
    "orange": "#f59f00",
    "purple": "#7b2cbf",
    "green": "#2d6a4f",
}


def rel(path: str | Path) -> Path:
    return ROOT / path


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(rel(path)).convert("RGB")
    return image


def add_image_panel(
    ax,
    path: str | Path,
    title: str,
    label: str,
    subtitle: str | None = None,
    fit: tuple[int, int] | None = None,
) -> None:
    image = load_image(path)
    if fit is not None:
        image = ImageOps.contain(image, fit, method=Image.Resampling.LANCZOS)
    ax.imshow(image)
    ax.set_axis_off()
    ax.text(
        0.015,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.88),
    )
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"], pad=8)
    if subtitle:
        ax.text(
            0.015,
            0.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.82),
        )


def save_figure(fig, stem: str) -> tuple[Path, Path]:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png, pdf


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with rel(path).open(newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def figure_01_overview() -> tuple[Path, Path]:
    fig = plt.figure(figsize=(12.5, 9.2), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.08)
    panels = [
        (
            grid[0, 0],
            "outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_preview_v001.png",
            "Materialized CT + vascular digital phantom",
            "A",
            "Blood/contrast material maps, organ labels, vessel wall, and embedded vascular fluid.",
        ),
        (
            grid[0, 1],
            "outputs/render3d/vascular_network_cleaned/ct_org_case0_imagetbad_case125_vascular_network_3d_render_preview_v001.png",
            "Transparent 3D phantom context",
            "B",
            "Body envelope, bone, lungs, liver, kidneys, arterial/venous lumen, and vessel wall.",
        ),
        (
            grid[1, 0],
            "outputs/sim/flow_4d_full_phantom_gif/ct_org_case0_imagetbad_case125_full_phantom_flow4d_velocity_contact_sheet_v001.png",
            "Pulsatile flow mapped into the phantom",
            "C",
            "Velocity-coded vascular graph frames across one cardiac cycle.",
        ),
        (
            grid[1, 1],
            "outputs/radiotherapy/dose_gamma_qa/ct_org_case0_imagetbad_case125_dose_gamma_qa_preview_v001.png",
            "Radiotherapy dose QA",
            "D",
            "Static reference dose compared against pulsatile vascular dose states with PyMedPhys gamma.",
        ),
    ]
    for spec, path, title, label, subtitle in panels:
        ax = fig.add_subplot(spec)
        add_image_panel(ax, path, title, label, subtitle)
    fig.suptitle(
        "Figure 1. Current digital phantom development state",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.955,
        "A CT-derived torso phantom now carries anatomical materials, a cleaned vascular network, coupled pulsatile flow, and radiotherapy QA outputs.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    return save_figure(fig, "figure_01_development_overview")


def figure_02_anatomy_atlas() -> tuple[Path, Path]:
    fig = plt.figure(figsize=(13.5, 8.4), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 4, hspace=0.18, wspace=0.03)
    views = [
        ("Anterior", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_anterior_view_v001.png"),
        ("Posterior", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_posterior_view_v001.png"),
        ("Left lateral", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_left_lateral_view_v001.png"),
        ("Right lateral", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_right_lateral_view_v001.png"),
        ("Superior", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_superior_view_v001.png"),
        ("Inferior", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_inferior_view_v001.png"),
        ("Oblique", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_oblique_view_v001.png"),
        ("Vascular zoom", "outputs/render3d/vascular_network_cleaned_atlas/views/ct_org_case0_imagetbad_case125_vascular_zoom_view_v001.png"),
    ]
    letters = "ABCDEFGH"
    for index, (title, path) in enumerate(views):
        ax = fig.add_subplot(grid[index // 4, index % 4])
        add_image_panel(ax, path, title, letters[index])
    fig.suptitle(
        "Figure 2. Multi-view 3D anatomy and vascular-network atlas",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.955,
        "Transparent tissue layers reveal bone, lungs, liver, kidneys, cleaned arterial/venous domains, and the synthetic vessel wall.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    return save_figure(fig, "figure_02_3d_anatomy_vascular_atlas")


def figure_03_flow_dynamics() -> tuple[Path, Path]:
    boundary_rows = read_csv("outputs/sim/flow_coupled_pulsatile/ct_org_case0_imagetbad_case125_coupled_pulsatile_boundary_timeseries_v001.csv")
    grouped = group_rows(boundary_rows, "node_id")
    fig = plt.figure(figsize=(13.5, 8.8), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 4, height_ratios=[1.08, 0.92], hspace=0.28, wspace=0.18)

    frame_indices = [0, 8, 16, 24]
    for i, frame_index in enumerate(frame_indices):
        ax = fig.add_subplot(grid[0, i])
        path = f"outputs/sim/flow_4d_full_phantom_gif/frames/ct_org_case0_imagetbad_case125_full_phantom_flow4d_velocity_frame_{frame_index:03d}.png"
        add_image_panel(ax, path, f"Phase {frame_index / 48:.2f}", chr(ord("A") + i))

    ax_flow = fig.add_subplot(grid[1, 0:2])
    for node_id, color, label, sign in [
        ("aorta_inlet", COLORS["red"], "Aorta inlet", 1.0),
        ("left_common_iliac_outlet", COLORS["orange"], "Left iliac outlet", -1.0),
        ("right_common_iliac_outlet", "#b45309", "Right iliac outlet", -1.0),
        ("left_renal_outlet", COLORS["green"], "Left renal outlet", -1.0),
        ("right_renal_outlet", "#40916c", "Right renal outlet", -1.0),
    ]:
        rows = grouped.get(node_id, [])
        if not rows:
            continue
        time = np.array([float(row["time_s"]) for row in rows])
        flow = sign * np.array([float(row["flow_ml_s"]) for row in rows])
        ax_flow.plot(time, flow, lw=2.2, color=color, label=label)
    ax_flow.set_title("E. Coupled pulsatile boundary flow", loc="left", fontsize=11, fontweight="bold")
    ax_flow.set_xlabel("Time over cardiac cycle (s)")
    ax_flow.set_ylabel("Flow magnitude (mL/s)")
    ax_flow.grid(True, color=COLORS["grid"], lw=0.7, alpha=0.8)
    ax_flow.legend(frameon=False, fontsize=8, ncol=2)

    ax_pressure = fig.add_subplot(grid[1, 2])
    for node_id, color, label in [
        ("aorta_inlet", COLORS["red"], "Aorta"),
        ("left_common_iliac_outlet", COLORS["orange"], "Left iliac"),
        ("left_renal_outlet", COLORS["green"], "Left renal"),
    ]:
        rows = grouped.get(node_id, [])
        if not rows:
            continue
        time = np.array([float(row["time_s"]) for row in rows])
        pressure = np.array([float(row["pressure_mmhg"]) for row in rows])
        ax_pressure.plot(time, pressure, lw=2.0, color=color, label=label)
    ax_pressure.set_title("F. Nodal pressure", loc="left", fontsize=11, fontweight="bold")
    ax_pressure.set_xlabel("Time (s)")
    ax_pressure.set_ylabel("Pressure (mmHg)")
    ax_pressure.grid(True, color=COLORS["grid"], lw=0.7, alpha=0.8)
    ax_pressure.legend(frameon=False, fontsize=8)

    ax_card = fig.add_subplot(grid[1, 3])
    ax_card.axis("off")
    cards = [
        ("Velocity range", "5.1-81.2 cm/s", COLORS["cyan"]),
        ("Rendered phases", "48 frames", COLORS["purple"]),
        ("Arterial inlet", "80 mL/s mean", COLORS["red"]),
        ("Mass balance", "< 1e-9 mL/s residual", COLORS["green"]),
    ]
    y = 0.92
    for title, value, color in cards:
        ax_card.add_patch(plt.Rectangle((0.04, y - 0.17), 0.92, 0.14, color=color, alpha=0.12, ec=color, lw=1.1))
        ax_card.text(0.08, y - 0.060, title, ha="left", va="center", fontsize=8.6, color=COLORS["muted"])
        ax_card.text(0.94, y - 0.115, value, ha="right", va="center", fontsize=10.0, fontweight="bold", color=COLORS["ink"])
        y -= 0.2
    ax_card.set_title("G. Flow QA snapshot", loc="left", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Figure 3. Pulsatile vascular-flow dynamics inside the phantom",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.955,
        "The coupled 1D flow solution is mapped onto the synthetic arterial/venous graph and rendered in the transparent anatomical context.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    return save_figure(fig, "figure_03_pulsatile_flow_dynamics")


def figure_04_rt_qa() -> tuple[Path, Path]:
    gamma_rows = read_csv("outputs/radiotherapy/dose_gamma_qa/ct_org_case0_imagetbad_case125_dose_gamma_qa_summary_v001.csv")
    dose_rows = read_csv("outputs/radiotherapy/planning_bundle/ct_org_case0_imagetbad_case125_rt_dose_metrics_v001.csv")
    target_rows = [row for row in dose_rows if row["mask_id"] == "target_ptv_synthetic_vertebral"]

    fig = plt.figure(figsize=(13.5, 8.8), facecolor=COLORS["paper"])
    grid = fig.add_gridspec(2, 3, width_ratios=[1.08, 1.08, 0.92], hspace=0.28, wspace=0.25)

    ax = fig.add_subplot(grid[0, 0:2])
    add_image_panel(
        ax,
        "outputs/radiotherapy/planning_bundle/ct_org_case0_imagetbad_case125_rt_planning_bundle_preview_v001.png",
        "Synthetic static and pulsatile RT dose states",
        "A",
    )
    ax = fig.add_subplot(grid[1, 0:2])
    add_image_panel(
        ax,
        "outputs/radiotherapy/dose_gamma_qa/ct_org_case0_imagetbad_case125_dose_gamma_qa_preview_v001.png",
        "PyMedPhys gamma QA and dose-difference preview",
        "B",
    )

    ax_gamma = fig.add_subplot(grid[0, 2])
    states = [row["state"].replace("pulsatile_", "") for row in gamma_rows]
    pass_rates = [float(row["pass_rate_percent"]) for row in gamma_rows]
    p95 = [float(row["p95_gamma"]) for row in gamma_rows]
    x = np.arange(len(states))
    bars = ax_gamma.bar(x, pass_rates, color=[COLORS["cyan"], COLORS["red"], COLORS["blue"]], alpha=0.86)
    ax_gamma.set_ylim(99.8, 100.03)
    ax_gamma.set_xticks(x)
    ax_gamma.set_xticklabels(states, rotation=20, ha="right")
    ax_gamma.set_ylabel("Gamma pass rate (%)")
    ax_gamma.set_title("C. 3%/3 mm gamma pass", loc="left", fontsize=11, fontweight="bold")
    ax_gamma.grid(True, axis="y", color=COLORS["grid"], lw=0.7)
    for rect, value, p95_value in zip(bars, pass_rates, p95):
        ax_gamma.text(rect.get_x() + rect.get_width() / 2, value + 0.003, f"{value:.1f}%\np95 {p95_value:.4f}", ha="center", va="bottom", fontsize=8)

    ax_dvh = fig.add_subplot(grid[1, 2])
    state_order = ["static", "pulsatile_mean", "pulsatile_peak", "pulsatile_trough"]
    label_map = {"static": "static", "pulsatile_mean": "mean", "pulsatile_peak": "peak", "pulsatile_trough": "trough"}
    d95 = []
    mean = []
    labels = []
    for state in state_order:
        match = next((row for row in target_rows if row["state"] == state), None)
        if match:
            labels.append(label_map[state])
            d95.append(float(match["d95_gy"]))
            mean.append(float(match["mean_dose_gy"]))
    x = np.arange(len(labels))
    ax_dvh.plot(x, d95, marker="o", lw=2.4, color=COLORS["red"], label="D95")
    ax_dvh.plot(x, mean, marker="s", lw=2.4, color=COLORS["blue"], label="Mean")
    ax_dvh.axhline(19.0, color=COLORS["grid"], lw=1.2, ls="--")
    ax_dvh.set_xticks(x)
    ax_dvh.set_xticklabels(labels, rotation=20, ha="right")
    ax_dvh.set_ylabel("Dose (Gy)")
    ax_dvh.set_title("D. Target PTV dose stability", loc="left", fontsize=11, fontweight="bold")
    ax_dvh.grid(True, color=COLORS["grid"], lw=0.7)
    ax_dvh.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Figure 4. Radiotherapy planning handoff and dose-gamma QA",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.955,
        "The phantom exports RT-ready structures/dose states and a PyMedPhys gamma workflow comparing static and pulsatile vascular conditions.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    return save_figure(fig, "figure_04_radiotherapy_gamma_qa")


def draw_box(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    ax.add_patch(
        plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=color, alpha=0.12, lw=1.4)
    )
    ax.add_patch(plt.Rectangle((x, y + h - 0.08), w, 0.08, facecolor=color, edgecolor=color, lw=0))
    ax.text(x + 0.025, y + h - 0.04, title, ha="left", va="center", fontsize=10.5, fontweight="bold", color="white")
    ax.text(
        x + 0.025,
        y + h - 0.11,
        textwrap.fill(body, width=23, break_long_words=True),
        ha="left",
        va="top",
        fontsize=7.8,
        color=COLORS["ink"],
        linespacing=1.28,
    )


def figure_05_pipeline() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(13.5, 6.8), facecolor=COLORS["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("Data", "CT-ORG anatomy and ImageTBAD vascular case staged as NIfTI source volumes.", COLORS["blue"]),
        ("Digital materials", "Organ labels converted into HU, mass density, RED, and tissue/material label maps.", COLORS["green"]),
        ("Vascular twin", "Synthetic arterial/venous graph voxelized into lumen, wall, flow-domain masks and meshes.", COLORS["red"]),
        ("Pulsatile flow", "Coupled 1D graph model produces pressure, flow, velocity and outlet split time series.", COLORS["orange"]),
        ("RT QA", "DICOM-RT-style CT/RTSTRUCT/RTDOSE handoff plus static-vs-pulsatile gamma and DVH metrics.", COLORS["purple"]),
    ]
    xs = np.linspace(0.035, 0.795, len(stages))
    for index, (title, body, color) in enumerate(stages):
        draw_box(ax, xs[index], 0.54, 0.17, 0.28, title, body, color)
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.006, 0.84),
                xytext=(xs[index] + 0.176, 0.84),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=COLORS["ink"]),
            )

    metrics = [
        ("Body volume", "23.13 L"),
        ("Vascular fluid", "133.99 cm3"),
        ("Vessel wall", "37.91 cm3"),
        ("Flow graph", "21 nodes / 19 edges"),
        ("Cardiac cycle", "160 samples"),
        ("Gamma QA", "100% pass sampled"),
    ]
    ax.text(0.035, 0.40, "Engineering snapshot", fontsize=13, fontweight="bold", color=COLORS["ink"])
    ax.text(0.035, 0.365, "Current quantitative build state from generated reports and manifests.", fontsize=9.5, color=COLORS["muted"])
    for index, (label, value) in enumerate(metrics):
        col = index % 3
        row = index // 3
        x = 0.035 + col * 0.305
        y = 0.23 - row * 0.13
        ax.add_patch(plt.Rectangle((x, y), 0.27, 0.09, facecolor="#f7f4ee", edgecolor="#e2ddd2", lw=1.0))
        ax.text(x + 0.018, y + 0.058, label, ha="left", va="center", fontsize=8.7, color=COLORS["muted"])
        ax.text(x + 0.252, y + 0.035, value, ha="right", va="center", fontsize=13, fontweight="bold", color=COLORS["ink"])

    ax.text(
        0.035,
        0.93,
        "Figure 5. Reproducible digital-phantom development workflow",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
    )
    ax.text(
        0.035,
        0.89,
        "The current build connects public CT/CTA-derived geometry, material maps, flow simulation, renderable meshes, and RT QA into one traceable digital twin.",
        fontsize=10.5,
        color=COLORS["muted"],
        ha="left",
    )
    return save_figure(fig, "figure_05_reproducible_pipeline_summary")


def write_captions(paths: list[tuple[str, Path, Path]]) -> None:
    captions = {
        "figure_01_development_overview": (
            "Figure 1. Current digital phantom development state. "
            "The generated panels summarize CT/material mapping, transparent 3D anatomy, pulsatile vascular-flow rendering, and RT gamma QA outputs."
        ),
        "figure_02_3d_anatomy_vascular_atlas": (
            "Figure 2. Multi-view 3D anatomy and vascular-network atlas. "
            "Standard views show the transparent phantom, organ context, bone, and cleaned arterial/venous vascular domains."
        ),
        "figure_03_pulsatile_flow_dynamics": (
            "Figure 3. Pulsatile vascular-flow dynamics inside the phantom. "
            "Velocity-colored frames and waveform plots show the coupled graph-flow model over one cardiac cycle."
        ),
        "figure_04_radiotherapy_gamma_qa": (
            "Figure 4. Radiotherapy planning handoff and dose-gamma QA. "
            "Synthetic RT dose states, PyMedPhys gamma analysis, and target dose stability are shown for static and pulsatile vascular conditions."
        ),
        "figure_05_reproducible_pipeline_summary": (
            "Figure 5. Reproducible digital-phantom development workflow. "
            "The engineering pipeline links staged public data, material maps, vascular flow, and RT QA into a traceable digital twin."
        ),
    }
    lines = ["# Publication Figure Captions", ""]
    for stem, png, pdf in paths:
        lines.extend(
            [
                f"## {stem}",
                "",
                captions[stem],
                "",
                f"- PNG: `{png}`",
                f"- PDF: `{pdf}`",
                "",
            ]
        )
    (OUT / "figure_captions.md").write_text("\n".join(lines))


def main() -> int:
    ensure_out()
    generated = [
        ("figure_01_development_overview", *figure_01_overview()),
        ("figure_02_3d_anatomy_vascular_atlas", *figure_02_anatomy_atlas()),
        ("figure_03_pulsatile_flow_dynamics", *figure_03_flow_dynamics()),
        ("figure_04_radiotherapy_gamma_qa", *figure_04_rt_qa()),
        ("figure_05_reproducible_pipeline_summary", *figure_05_pipeline()),
    ]
    write_captions(generated)
    print("Generated publication figures:")
    for stem, png, pdf in generated:
        print(f"- {stem}: {png} | {pdf}")
    print(f"- captions: {OUT / 'figure_captions.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
