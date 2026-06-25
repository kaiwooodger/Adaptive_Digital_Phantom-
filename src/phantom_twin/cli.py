from __future__ import annotations

import argparse
from pathlib import Path

from .anthropometry import (
    build_anthropometric_torso_morph,
    format_anthropometric_torso_morph_result,
)
from .aorta_registration_benchmark import (
    apply_learned_aorta_to_vascular_graph,
    build_aorta_registration_benchmark,
    format_aorta_registration_benchmark_result,
    format_learned_aorta_graph_result,
)
from .avt_kits_aorta import format_avt_kits_aorta_result, stage_avt_kits_aorta_zip
from .btcv_branch_anchor_graph import (
    build_btcv_branch_anchor_vascular_graph,
    format_btcv_branch_anchor_graph_result,
)
from .btcv_ivc_tighten import format_btcv_ivc_tightening_result, tighten_btcv_ivc_trunk
from .combined import (
    build_combined_digital_phantom,
    format_combined_digital_phantom_result,
)
from .coarse_vessel_graph import build_btcv_coarse_vessel_graph, format_coarse_vessel_graph_result
from .corrected_branch_status import (
    build_corrected_branch_status_report,
    format_corrected_branch_status_result,
)
from .corrected_branch_release import (
    build_corrected_branch_release_package,
    format_corrected_branch_release_result,
)
from .corrected_branch_audit import (
    audit_corrected_branch_release_package,
    format_corrected_branch_release_audit_result,
)
from .ct_org_cohort import format_ct_org_label_cohort_result, stage_ct_org_label_cohort
from .cta_vascular_graph import build_cta_derived_vascular_graph, format_cta_derived_vascular_graph_result
from .datasets import load_dataset_manifest, summarize_datasets
from .deformation_match_experiment import (
    format_deformation_match_experiment_result,
    run_deformation_match_experiment,
)
from .dose_gamma_qa import build_dose_gamma_qa, format_dose_gamma_qa_result
from .flow_boundary import (
    build_flow_boundary_package,
    format_flow_boundary_package_result,
)
from .flow_1d import build_flow_1d_model, format_flow_1d_model_result
from .flow_coupled import (
    build_coupled_pulsatile_flow_model,
    format_coupled_pulsatile_flow_result,
)
from .flow_pulsatile import build_pulsatile_flow_model, format_pulsatile_flow_result
from .flow_visualization import (
    build_4d_flow_visualization,
    format_4d_flow_visualization_result,
)
from .gi_autoseg_bridge import auto_stage_gi_segmentation, format_gi_autoseg_bridge_result
from .gi_segmentation_staging import format_gi_segmentation_staging_result, stage_gi_segmentation
from .cartridge import (
    build_printable_vascular_cartridge,
    format_printable_cartridge_result,
)
from .labeled_vessel_graph import (
    build_labeled_vessel_vascular_graph,
    build_registered_labeled_vessel_vascular_graph,
    format_labeled_vessel_graph_result,
    format_medseg_vascular_staging_result,
    format_registered_labeled_vessel_graph_result,
    stage_medseg_abdominal_vasculature,
)
from .label_vessel_flow_domain import (
    build_label_vessel_flow_domain,
    format_label_vessel_flow_domain_result,
)
from .materials import load_material_library, summarize_materials
from .mesh_clean import (
    MeshCleanConfig,
    clean_meshes,
    format_cleaning_report,
    write_cleaning_report,
)
from .mesh_qa import analyze_meshes, format_mesh_qa_markdown, write_mesh_qa_report
from .meshes import export_label_mesh
from .nifti import format_nifti_summary, inspect_nifti
from .pca_mode_qa import format_pca_mode_qa_result, rank_pca_modes
from .pca_phantom_set import build_approved_pca_phantom_set, format_approved_pca_phantom_set_result
from .pca_variants import generate_pca_mode_variants, format_pca_mode_variant_atlas_result
from .patient_adapter import build_patient_phantom_adapter, format_patient_phantom_adapter_result
from .patient_build import format_patient_phantom_build_result, run_patient_phantom_build
from .patient_build_qa import format_patient_build_qa_result, qa_patient_phantom_build
from .patient_case_adapter import format_patient_case_adapter_result, run_patient_case_adapter
from .phantom_experiment import format_phantom_experiment_set_result, run_phantom_experiment_set
from .population_cohort import build_population_cohort, format_population_cohort_result
from .product_case_runner import build_product_case, format_product_case_result
from .product_release import build_product_release_package, format_product_release_result
from .product_release_case_runner import build_product_release_case, format_product_release_case_result
from .profile_comparison import build_profile_rerun_comparison_atlas, format_profile_rerun_comparison_result
from .profile_adapter import build_user_profile_adapter, format_user_profile_adapter_result
from .profile_envelope import build_profile_operating_envelope, format_profile_envelope_result
from .profile_planner import format_profile_planning_result, plan_next_profile_validations
from .profile_prescription import build_profile_operating_prescription, format_profile_prescription_result
from .profile_sweep import build_profile_sweep, format_profile_sweep_result
from .render3d import (
    generate_3d_view_atlas,
    generate_combined_3d_render,
    generate_vascular_network_3d_render,
    format_render_atlas_result,
    format_render3d_result,
)
from .radiotherapy import (
    build_radiotherapy_qa_package,
    format_radiotherapy_qa_package_result,
)
from .reg_training_testing import format_reg_training_testing_staging_result, stage_reg_training_testing_zip
from .reg_training_benchmark import (
    build_reg_training_testing_benchmark,
    format_reg_training_testing_benchmark_result,
)
from .registration_anchor_qa import (
    RegistrationAnchorThresholds,
    format_registration_anchor_qa_result,
    rank_registration_anchors,
)
from .reports import build_phase1_summary, load_phase1_config
from .release_readiness import audit_research_release_package, format_release_readiness_audit_result
from .release_package import build_research_release_package, format_research_release_package_result
from .research_demonstrator import build_research_demonstrator_package, format_research_demonstrator_result
from .stage007_baseline import (
    format_stage007_acceptance_smoke_result,
    format_stage007_baseline_promotion_result,
    promote_stage007_clean_baseline,
    run_stage007_acceptance_smoke,
)
from .rt_planning import (
    analyze_spatial_rt_flow_coupling,
    build_rt_planning_bundle,
    build_spatial_rt_flow_dose_model,
    format_rt_planning_bundle_result,
    format_spatial_rt_flow_coupling_result,
    format_spatial_rt_flow_dose_result,
)
from .rt_flow_model_compare import (
    compare_scalar_vs_spatial_rt_flow_dose,
    format_rt_flow_model_comparison_result,
)
from .statistical_anatomy import (
    build_statistical_anatomy_morph,
    format_statistical_anatomy_morph_result,
)
from .status_atlas import build_current_phantom_status_atlas, format_current_phantom_status_atlas_result
from .torso import build_digital_torso_phantom, format_digital_torso_result
from .validation_intake import build_validation_intake_package, format_validation_intake_result
from .validation_discovery import discover_validation_candidates, format_validation_discovery_result
from .validation_case_staging import format_validation_case_staging_result, stage_validation_case
from .validation_case_promotion import format_validation_case_promotion_result, promote_harmonized_vessel_case
from .validation_roadmap import build_validation_roadmap, format_validation_roadmap_result
from .variant_rerun_harness import build_variant_rerun_harness, format_variant_rerun_harness_result
from .variant_rt_compare import compare_variant_rt_impact, format_variant_rt_comparison_result
from .vascular_domain_connectivity import (
    format_vascular_domain_connectivity_repair_result,
    repair_vascular_domain_connectivity,
)
from .vascular import (
    design_vascular_flow_loop,
    format_flow_loop_result,
    format_vascular_result,
    prepare_vascular_module,
)
from .vascular_network import (
    build_vascular_network_scaffold,
    deform_vascular_graph_for_variant,
    format_variant_graph_deformation_result,
    format_vascular_network_scaffold_result,
)
from .vessel_anatomy_validation import (
    format_vessel_anatomy_validation_result,
    validate_vessel_organ_anatomy,
)
from .vessel_anatomy_correction import (
    correct_vessel_bone_conflicts,
    format_vessel_anatomy_correction_result,
)
from .vessel_edge_reroute import (
    format_vessel_edge_reroute_result,
    reroute_vessel_edge_around_bone,
)
from .vessel_ctgrid_resample import format_vessel_ct_grid_resample_result, resample_vessel_to_ct_grid
from .vessel_label_anatomy_qa import format_vessel_label_anatomy_qa_result, qa_vessel_label_anatomy
from .vessel_label_anatomy_correction import format_vessel_label_anatomy_correction_result, correct_vessel_label_anatomy
from .vessel_label_harmonizer import format_vessel_label_harmonization_result, harmonize_vessel_labels
from .vessel_radius_validation import (
    format_vessel_radius_validation_result,
    validate_vessel_radius_anatomy,
)
from .vessel_radius_tuning import (
    format_vessel_radius_tuning_result,
    tune_vessel_radii_against_bone,
)
from .vessel_outside_body_repair import (
    format_vessel_outside_body_repair_result,
    repair_vessel_outside_body_margin,
)
from .vascular_voxelize import (
    format_vascular_network_voxelization_result,
    voxelize_vascular_network,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATERIALS = PROJECT_ROOT / "configs" / "materials.yaml"
DEFAULT_DATASETS = PROJECT_ROOT / "configs" / "datasets.yaml"
DEFAULT_PHASE1 = PROJECT_ROOT / "configs" / "phase1_torso_mvp.yaml"
DEFAULT_CT_ORG_LABELMAP = PROJECT_ROOT / "configs" / "labelmaps" / "ct_org.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phantom-twin",
        description="Phase 1 helpers for the phantom digital-twin project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    materials = subparsers.add_parser("materials-check", help="Validate and summarize material targets.")
    materials.add_argument("--materials", default=str(DEFAULT_MATERIALS))

    datasets = subparsers.add_parser("datasets-list", help="Validate and summarize dataset sources.")
    datasets.add_argument("--datasets", default=str(DEFAULT_DATASETS))

    phase1 = subparsers.add_parser("phase1-summary", help="Summarize the Phase 1 MVP configuration.")
    phase1.add_argument("--phase1", default=str(DEFAULT_PHASE1))
    phase1.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    phase1.add_argument("--datasets", default=str(DEFAULT_DATASETS))

    inspect = subparsers.add_parser("inspect-nifti", help="Inspect a NIfTI image or label volume.")
    inspect.add_argument("path")

    mesh = subparsers.add_parser("export-label-mesh", help="Export one integer label from a NIfTI mask to a mesh.")
    mesh.add_argument("--labels", required=True, help="Path to label NIfTI file.")
    mesh.add_argument("--label-id", required=True, type=int, nargs="+")
    mesh.add_argument("--output", required=True, help="Output mesh path, usually .stl.")
    mesh.add_argument("--level", default=0.5, type=float)

    mesh_qa = subparsers.add_parser("mesh-qa", help="Analyze mesh quality and write a Markdown report.")
    mesh_qa.add_argument("meshes", nargs="+", help="Mesh files to analyze.")
    mesh_qa.add_argument("--output", default=None, help="Optional Markdown report output path.")

    clean = subparsers.add_parser("clean-meshes", help="Clean STL/PLY meshes for CAD preparation.")
    clean.add_argument("meshes", nargs="+", help="Mesh files to clean.")
    clean.add_argument("--output-dir", default="outputs/cad/cleaned_meshes")
    clean.add_argument("--formats", nargs="+", default=["stl", "ply"], help="Output mesh formats.")
    clean.add_argument("--suffix", default="cleaned_v001")
    clean.add_argument("--min-component-faces", type=int, default=100)
    clean.add_argument("--target-max-faces", type=int, default=80_000)
    clean.add_argument(
        "--fill-holes",
        choices=["none", "single", "fan"],
        default="single",
        help="Hole filling strategy. 'single' is conservative; 'fan' can cap larger holes.",
    )
    clean.add_argument("--report", default=None, help="Optional cleaning Markdown report.")
    clean.add_argument("--qa-report", default=None, help="Optional QA report for cleaned outputs.")

    vascular = subparsers.add_parser(
        "prepare-vascular-module",
        help="Prepare smoothed lumen, centerline, and port planes from a vascular label map.",
    )
    vascular.add_argument("--labels", required=True, help="Path to vascular label NIfTI file.")
    vascular.add_argument("--label-id", type=int, default=1, help="Label ID to use as the lumen.")
    vascular.add_argument("--case-id", default="imagetbad_case125_true_lumen")
    vascular.add_argument("--output-dir", default="outputs/cad/vascular_module")
    vascular.add_argument("--smooth-sigma", type=float, default=1.0)
    vascular.add_argument("--target-max-faces", type=int, default=60_000)
    vascular.add_argument("--centerline-method", choices=["skeleton", "axial"], default="skeleton")
    vascular.add_argument("--formats", nargs="+", default=["stl", "ply"])
    vascular.add_argument("--report", default="outputs/reports/vascular_module_stage001.md")
    vascular.add_argument("--qa-report", default="outputs/reports/vascular_module_mesh_qa_stage001.md")

    flow_loop = subparsers.add_parser(
        "design-flow-loop",
        help="Create first-pass vascular port adapter and flow-loop reference geometry.",
    )
    flow_loop.add_argument("--ports", required=True, help="Path to vascular port YAML.")
    flow_loop.add_argument("--lumen-mesh", required=True, help="Path to smoothed lumen mesh.")
    flow_loop.add_argument("--output-dir", default="outputs/cad/vascular_flow_loop")
    flow_loop.add_argument("--formats", nargs="+", default=["stl", "ply"])
    flow_loop.add_argument("--wall-thickness-mm", type=float, default=2.0)
    flow_loop.add_argument("--sleeve-length-mm", type=float, default=24.0)
    flow_loop.add_argument("--barb-count", type=int, default=2)
    flow_loop.add_argument("--barb-height-mm", type=float, default=1.0)
    flow_loop.add_argument("--barb-width-mm", type=float, default=2.0)
    flow_loop.add_argument("--barb-spacing-mm", type=float, default=5.0)
    flow_loop.add_argument("--flange-thickness-mm", type=float, default=4.0)
    flow_loop.add_argument("--flange-extra-radius-mm", type=float, default=5.0)
    flow_loop.add_argument("--pressure-tap-diameter-mm", type=float, default=3.0)
    flow_loop.add_argument("--report", default="outputs/reports/vascular_flow_loop_stage001.md")
    flow_loop.add_argument("--qa-report", default="outputs/reports/vascular_flow_loop_mesh_qa_stage001.md")

    network = subparsers.add_parser(
        "build-vascular-network-scaffold",
        help="Create a synthetic major-vessel graph scaffold with tube meshes and metadata.",
    )
    network.add_argument("--combined-spec", required=True, help="Combined digital phantom spec YAML.")
    network.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    network.add_argument("--output-dir", default="outputs/digital/vascular_network")
    network.add_argument("--formats", nargs="+", default=["stl", "ply", "obj"])
    venous = network.add_mutually_exclusive_group()
    venous.add_argument(
        "--include-venous-return",
        dest="include_venous_return",
        action="store_true",
        default=True,
        help="Include the optional synthetic venous return path. This is the default.",
    )
    venous.add_argument(
        "--skip-venous-return",
        dest="include_venous_return",
        action="store_false",
        help="Build only the arterial scaffold.",
    )
    network.add_argument("--body-mesh", default=None, help="Optional body-envelope mesh for preview context.")
    network.add_argument("--report", default="outputs/reports/vascular_network_scaffold_stage001.md")
    network.add_argument("--qa-report", default="outputs/reports/vascular_network_scaffold_mesh_qa_stage001.md")

    cta_graph = subparsers.add_parser(
        "build-cta-derived-vascular-graph",
        help="Replace the synthetic aorta trunk with a CTA-mask-derived centerline while retaining branch placeholders.",
    )
    cta_graph.add_argument("--baseline-graph", required=True, help="Existing vascular graph YAML to update.")
    cta_graph.add_argument("--vascular-mask", required=True, help="CTA-derived vascular lumen mask NIfTI.")
    cta_graph.add_argument("--case-id", default="ct_org_case0_imagetbad_case125_cta_derived")
    cta_graph.add_argument("--output-dir", default="outputs/digital/vascular_network_cta_derived")
    cta_full_trunk = cta_graph.add_mutually_exclusive_group()
    cta_full_trunk.add_argument(
        "--coarse-aorta-full-trunk",
        dest="coarse_aorta_full_trunk",
        action="store_true",
        default=True,
        help="For coarse one-edge aorta graphs, use the full CTA trunk extent. This is the default.",
    )
    cta_full_trunk.add_argument(
        "--no-coarse-aorta-full-trunk",
        dest="coarse_aorta_full_trunk",
        action="store_false",
        help="Only replace the graph endpoint-to-endpoint aorta segment.",
    )
    cta_graph.add_argument("--report", default="outputs/reports/cta_derived_vascular_graph_stage002.md")

    aorta_benchmark = subparsers.add_parser(
        "build-aorta-registration-benchmark",
        help="Build a leave-one-out registration benchmark and learned aorta model from staged aorta masks.",
    )
    aorta_benchmark.add_argument("--manifest-csv", required=True, help="Manifest CSV with aorta_mask_nifti_path or vessel_seg_path.")
    aorta_benchmark.add_argument("--dataset-id", default="avt_kits_aorta_benchmark_stage001")
    aorta_benchmark.add_argument("--output-dir", default="outputs/digital/aorta_registration_benchmark")
    aorta_benchmark.add_argument("--sample-count", type=int, default=64)
    aorta_benchmark.add_argument("--label-value", type=int, default=1)
    aorta_benchmark.add_argument("--use-any-nonzero-label", action="store_true")
    aorta_benchmark.add_argument("--report", default="outputs/reports/aorta_registration_benchmark_stage001.md")

    learned_aorta_graph = subparsers.add_parser(
        "apply-learned-aorta-graph",
        help="Replace a vascular graph aorta trunk with a population-learned aorta model registered to graph endpoints.",
    )
    learned_aorta_graph.add_argument("--graph", required=True, help="Source vascular graph YAML.")
    learned_aorta_graph.add_argument("--aorta-model", required=True, help="Aorta benchmark model YAML or NPZ.")
    learned_aorta_graph.add_argument("--case-id", default="population_learned_aorta_graph")
    learned_aorta_graph.add_argument("--output-dir", default="outputs/digital/vascular_network_learned_aorta")
    learned_aorta_graph.add_argument("--source-node-id", default="aorta_inlet")
    learned_aorta_graph.add_argument("--target-node-id", default="descending_aorta_mid")
    learned_aorta_graph.add_argument("--edge-flow-role", default="aorta_trunk")
    learned_aorta_graph.add_argument("--point-count", type=int, default=64)
    learned_aorta_graph.add_argument("--radius-scale", type=float, default=1.0)
    learned_aorta_graph.add_argument("--max-radius-mm", type=float, default=None)
    learned_aorta_graph.add_argument(
        "--minimum-target-span-mm",
        type=float,
        default=None,
        help="Optional lower bound for learned-aorta endpoint span; prevents endpoint-compressed trunks.",
    )
    learned_aorta_graph.add_argument("--report", default="outputs/reports/population_learned_aorta_graph_stage001.md")

    medseg_stage = subparsers.add_parser(
        "stage-medseg-abdominal-vasculature",
        help="Validate and stage the MedSeg branch-rich abdominal vasculature teaching case.",
    )
    medseg_stage.add_argument("--raw-dir", default="data/raw/medseg_vasculature_abdomen")
    medseg_stage.add_argument("--label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    medseg_stage.add_argument("--case-id", default="medseg_abdominal_vasculature_case001")
    medseg_stage.add_argument("--output-dir", default="data/processed/medseg_vasculature_abdomen")
    medseg_stage.add_argument("--report", default="outputs/reports/medseg_abdominal_vasculature_staging_stage001.md")

    labeled_graph = subparsers.add_parser(
        "build-labeled-vessel-vascular-graph",
        help="Replace vascular placeholder edge polylines using branch-rich labeled vessel centerline templates.",
    )
    labeled_graph.add_argument("--baseline-graph", required=True, help="Existing CTA/synthetic vascular graph YAML.")
    labeled_graph.add_argument("--labeled-mask", required=True, help="Branch-rich labeled vessel mask NIfTI.")
    labeled_graph.add_argument("--label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    labeled_graph.add_argument("--case-id", default="ct_org_case0_medseg_branch_template")
    labeled_graph.add_argument("--output-dir", default="outputs/digital/vascular_network_medseg_branch_template")
    labeled_graph.add_argument("--report", default="outputs/reports/medseg_branch_template_vascular_graph_stage001.md")

    registered_labeled_graph = subparsers.add_parser(
        "build-registered-labeled-vessel-vascular-graph",
        help="Register branch-rich labelled vessel centerlines to a target phantom graph using vascular landmarks.",
    )
    registered_labeled_graph.add_argument("--target-graph", required=True, help="Target phantom/PCA vascular graph YAML.")
    registered_labeled_graph.add_argument("--labeled-mask", required=True, help="Branch-rich labelled vessel mask NIfTI.")
    registered_labeled_graph.add_argument("--label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    registered_labeled_graph.add_argument("--target-labels", default=None, help="Optional target anatomy material-label NIfTI for provenance.")
    registered_labeled_graph.add_argument("--case-id", default="ct_org_mode03_neg_medseg_registered")
    registered_labeled_graph.add_argument("--output-dir", default="outputs/digital/vascular_network_medseg_registered")
    registered_labeled_graph.add_argument("--report", default="outputs/reports/medseg_registered_vascular_graph_stage001.md")

    btcv_branch_anchor = subparsers.add_parser(
        "build-btcv-branch-anchor-vascular-graph",
        help="Rebuild a coarse BTCV aorta/IVC graph with branch anchor nodes for MedSeg/CTA branch registration.",
    )
    btcv_branch_anchor.add_argument("--coarse-graph", required=True, help="Bone-corrected coarse BTCV vascular graph YAML.")
    btcv_branch_anchor.add_argument("--anatomy-labels", required=True, help="BTCV material/anatomy labels used to estimate organ branch anchors.")
    btcv_branch_anchor.add_argument("--case-id", default="btcv_branch_anchor_vascular_graph")
    btcv_branch_anchor.add_argument("--output-dir", default="outputs/digital/btcv_branch_anchor_graph")
    btcv_branch_anchor.add_argument("--report", default="outputs/reports/btcv_branch_anchor_vascular_graph.md")

    btcv_ivc_tighten = subparsers.add_parser(
        "tighten-btcv-ivc-trunk",
        help="Replace duplicated full-IVC registered curves with compact local IVC trunk segments inside the BTCV body.",
    )
    btcv_ivc_tighten.add_argument("--graph", required=True, help="Registered branch-rich BTCV vascular graph YAML.")
    btcv_ivc_tighten.add_argument("--anatomy-labels", required=True, help="BTCV material/anatomy labels used for body clipping.")
    btcv_ivc_tighten.add_argument("--case-id", default="btcv_ivc_tightened")
    btcv_ivc_tighten.add_argument("--output-dir", default="outputs/digital/btcv_ivc_tightened")
    btcv_ivc_tighten.add_argument("--point-count", type=int, default=36)
    btcv_ivc_tighten.add_argument("--report", default="outputs/reports/btcv_ivc_tightening.md")

    graph_deform = subparsers.add_parser(
        "deform-vascular-graph-for-variant",
        help="Deform the baseline vascular graph into a PCA/anatomy variant before voxelized flow rerun.",
    )
    graph_deform.add_argument("--baseline-graph", required=True, help="Baseline vascular graph YAML.")
    graph_deform.add_argument("--baseline-labels", required=True, help="Baseline material-label NIfTI.")
    graph_deform.add_argument("--variant-labels", required=True, help="Variant material-label NIfTI.")
    graph_deform.add_argument("--variant-id", default="mode01_pos")
    graph_deform.add_argument("--case-id", default="ct_org_label_population8_pca_modes_stage001_mode01_pos")
    graph_deform.add_argument("--output-dir", default="outputs/digital/variant_vascular_graph")
    graph_deform.add_argument("--report", default="outputs/reports/variant_vascular_graph_deformation_stage001.md")

    voxelize_network = subparsers.add_parser(
        "voxelize-vascular-network",
        help="Voxelize a vascular graph scaffold into NIfTI lumen, wall, boundary, and material-label maps.",
    )
    voxelize_network.add_argument("--graph", required=True, help="Vascular network graph YAML.")
    voxelize_network.add_argument("--combined-labels", required=True, help="Combined material-label NIfTI to paint into.")
    voxelize_network.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    voxelize_network.add_argument("--source-ct", default=None, help="Optional CT NIfTI for preview background.")
    voxelize_network.add_argument("--body-mask", default=None, help="Optional body-mask NIfTI for clipping.")
    voxelize_network.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    voxelize_network.add_argument("--output-dir", default="outputs/digital/vascular_network_voxelized")
    voxelize_network.add_argument("--sample-step-mm", type=float, default=0.75)
    voxelize_network.add_argument("--vessel-wall-thickness-mm", type=float, default=2.0)
    voxelize_network.add_argument(
        "--contrast-mode",
        choices=["arterial", "all", "none"],
        default="arterial",
        help="Which voxelized lumen should be marked contrast-filled in the contrast material map.",
    )
    voxelize_network.add_argument(
        "--collision-cleanup",
        choices=["nearest-centerline", "arterial-priority", "venous-priority", "none"],
        default="nearest-centerline",
        help="How to assign voxels where arterial and venous tube masks overlap.",
    )
    clipping = voxelize_network.add_mutually_exclusive_group()
    clipping.add_argument(
        "--clip-to-body",
        dest="clip_to_body",
        action="store_true",
        default=True,
        help="Clip voxelized network to the supplied body mask or non-air combined labels. This is the default.",
    )
    clipping.add_argument(
        "--no-clip-to-body",
        dest="clip_to_body",
        action="store_false",
        help="Allow voxelized network outside the body mask.",
    )
    material_volumes = voxelize_network.add_mutually_exclusive_group()
    material_volumes.add_argument(
        "--write-material-volumes",
        dest="write_material_volumes",
        action="store_true",
        default=True,
        help="Write dense HU, density, and RED volumes. This is the default.",
    )
    material_volumes.add_argument(
        "--skip-material-volumes",
        dest="write_material_volumes",
        action="store_false",
        help="Skip dense material-property volumes for disk-light flow reruns.",
    )
    voxelize_network.add_argument("--report", default="outputs/reports/vascular_network_voxelized_stage001.md")

    coarse_btcv_graph = subparsers.add_parser(
        "build-btcv-coarse-vessel-graph",
        help="Prune a branch-rich graph to the BTCV-supported coarse aorta/IVC major-vessel product domain.",
    )
    coarse_btcv_graph.add_argument("--graph-yaml", required=True, help="Source branch-rich vascular graph YAML.")
    coarse_btcv_graph.add_argument("--case-id", default="btcv_coarse_major_vessels")
    coarse_btcv_graph.add_argument("--output-dir", default="outputs/digital/btcv_coarse_vessel_graph")
    coarse_btcv_graph.add_argument("--arterial-radius-mm", type=float, default=2.5)
    coarse_btcv_graph.add_argument("--venous-radius-mm", type=float, default=3.0)
    coarse_btcv_graph.add_argument("--report", default="outputs/reports/btcv_coarse_vessel_graph.md")

    vessel_anatomy = subparsers.add_parser(
        "validate-vessel-organ-anatomy",
        help="Validate registered vascular graph and voxelized lumen/wall masks against organ/body material labels.",
    )
    vessel_anatomy.add_argument("--voxelized-spec", required=True, help="Voxelized vascular-network spec YAML.")
    vessel_anatomy.add_argument("--graph", default=None, help="Optional graph YAML override. Defaults to the graph in the voxelized spec.")
    vessel_anatomy.add_argument(
        "--anatomy-labels",
        default=None,
        help="Optional material-label NIfTI override. Defaults to the pre-vascular combined labels in the voxelized spec.",
    )
    vessel_anatomy.add_argument("--case-id", default="ct_org_vessel_organ_validation")
    vessel_anatomy.add_argument("--output-dir", default="outputs/validation/vessel_organ_anatomy")
    vessel_anatomy.add_argument("--sample-step-mm", type=float, default=2.0)
    vessel_anatomy.add_argument("--report", default="outputs/reports/vessel_organ_anatomy_validation_stage001.md")

    vessel_correction = subparsers.add_parser(
        "correct-vessel-bone-conflicts",
        help="Move vascular graph nodes and edge centerlines out of bone while preserving IDs and flow roles.",
    )
    vessel_correction.add_argument("--graph", required=True, help="Vascular graph YAML to correct.")
    vessel_correction.add_argument("--anatomy-labels", required=True, help="Material-label NIfTI used for body/bone correction fields.")
    vessel_correction.add_argument(
        "--edge-metrics",
        default=None,
        help="Optional vessel-organ edge metrics CSV; reviewed/bone-intersecting edges are targeted.",
    )
    vessel_correction.add_argument("--case-id", default="ct_org_vessel_bone_corrected")
    vessel_correction.add_argument("--output-dir", default="outputs/digital/vessel_anatomy_corrected")
    vessel_correction.add_argument("--clearance-mm", type=float, default=8.0)
    vessel_correction.add_argument("--edge-bone-review-threshold", type=float, default=0.05)
    vessel_correction.add_argument("--max-node-shift-mm", type=float, default=24.0)
    vessel_correction.add_argument("--max-point-shift-mm", type=float, default=24.0)
    vessel_correction.add_argument("--smooth-iterations", type=int, default=1)
    vessel_correction.add_argument("--report", default="outputs/reports/vessel_anatomy_correction_stage001.md")

    vessel_reroute = subparsers.add_parser(
        "reroute-vessel-edge-around-bone",
        help="Reroute one vascular graph edge around bone while preserving topology and vessel metadata.",
    )
    vessel_reroute.add_argument("--graph", required=True, help="Vascular graph YAML to reroute.")
    vessel_reroute.add_argument("--anatomy-labels", required=True, help="Material-label NIfTI used for body/bone fields.")
    vessel_reroute.add_argument("--edge-id", required=True, help="Edge ID to reroute.")
    vessel_reroute.add_argument("--case-id", default="targeted_vessel_edge_reroute")
    vessel_reroute.add_argument("--output-dir", default="outputs/digital/vessel_edge_reroutes")
    vessel_reroute.add_argument("--clearance-mm", type=float, default=8.0)
    vessel_reroute.add_argument("--max-detour-mm", type=float, default=90.0)
    vessel_reroute.add_argument("--detour-step-mm", type=float, default=6.0)
    vessel_reroute.add_argument("--sample-step-mm", type=float, default=2.0)
    vessel_reroute.add_argument("--resample-step-mm", type=float, default=3.0)
    vessel_reroute.add_argument("--max-point-shift-mm", type=float, default=18.0)
    vessel_reroute.add_argument("--smooth-iterations", type=int, default=2)
    vessel_reroute.add_argument("--report", default="outputs/reports/vessel_edge_reroute_stage001.md")

    outside_body_repair = subparsers.add_parser(
        "repair-vessel-outside-body-margin",
        help="Trim endpoint vessels and add local radius profiles so tube masks fit inside the body before clipping.",
    )
    outside_body_repair.add_argument("--graph", required=True, help="Vascular graph YAML to repair.")
    outside_body_repair.add_argument("--anatomy-labels", required=True, help="Material-label NIfTI used as the body envelope.")
    outside_body_repair.add_argument("--edge-ids", nargs="*", default=(), help="Optional explicit edge IDs. Defaults to all edges with outside-body voxels.")
    outside_body_repair.add_argument("--case-id", default="vessel_outside_body_repaired")
    outside_body_repair.add_argument("--output-dir", default="outputs/digital/vessel_outside_body_repair")
    outside_body_repair.add_argument("--sample-step-mm", type=float, default=0.9)
    outside_body_repair.add_argument("--body-margin-mm", type=float, default=0.75)
    outside_body_repair.add_argument("--min-radius-mm", type=float, default=0.5)
    outside_body_repair.add_argument("--max-profile-points", type=int, default=128)
    outside_body_repair.add_argument("--report", default="outputs/reports/vessel_outside_body_repair.md")

    radius_validation = subparsers.add_parser(
        "validate-vessel-radius-anatomy",
        help="Evaluate vessel-organ clearance using graph tube radii instead of centerline samples only.",
    )
    radius_validation.add_argument("--voxelized-spec", required=True, help="Voxelized vascular-network spec YAML.")
    radius_validation.add_argument("--graph", default=None, help="Optional graph YAML override. Defaults to the graph in the voxelized spec.")
    radius_validation.add_argument(
        "--anatomy-labels",
        default=None,
        help="Optional material-label NIfTI override. Defaults to the pre-vascular combined labels in the voxelized spec.",
    )
    radius_validation.add_argument("--case-id", default="ct_org_vessel_radius_validation")
    radius_validation.add_argument("--output-dir", default="outputs/validation/vessel_radius_anatomy")
    radius_validation.add_argument("--sample-step-mm", type=float, default=2.0)
    radius_validation.add_argument("--scaled-radius-factor", type=float, default=0.75)
    radius_validation.add_argument("--review-lumen-bone-fraction", type=float, default=0.10)
    radius_validation.add_argument("--fail-lumen-bone-fraction", type=float, default=0.35)
    radius_validation.add_argument("--report", default="outputs/reports/vessel_radius_anatomy_validation_stage001.md")

    radius_tuning = subparsers.add_parser(
        "tune-vessel-radii-against-bone",
        help="Add anatomy-aware radius profiles to vessel edges with tube-volume overlap near bone.",
    )
    radius_tuning.add_argument("--graph", required=True, help="Vascular graph YAML to tune.")
    radius_tuning.add_argument("--anatomy-labels", required=True, help="Material-label NIfTI used for bone distance fields.")
    radius_tuning.add_argument("--radius-metrics", default=None, help="Optional radius-aware edge metrics CSV; review edges are tuned.")
    radius_tuning.add_argument("--edge-ids", nargs="*", default=(), help="Optional explicit edge IDs to tune.")
    radius_tuning.add_argument("--case-id", default="vessel_radius_tuned")
    radius_tuning.add_argument("--output-dir", default="outputs/digital/vessel_radius_tuned")
    radius_tuning.add_argument("--bone-clearance-mm", type=float, default=0.5)
    radius_tuning.add_argument("--sample-step-mm", type=float, default=2.0)
    radius_tuning.add_argument("--max-profile-points", type=int, default=56)
    radius_tuning.add_argument("--smooth-iterations", type=int, default=2)
    radius_tuning.add_argument("--min-radius-mm", type=float, default=1.5)
    radius_tuning.add_argument("--branch-max-radius-mm", type=float, default=8.0)
    radius_tuning.add_argument("--arterial-trunk-min-radius-mm", type=float, default=4.0)
    radius_tuning.add_argument("--arterial-trunk-max-radius-mm", type=float, default=18.0)
    radius_tuning.add_argument("--venous-trunk-min-radius-mm", type=float, default=3.0)
    radius_tuning.add_argument("--venous-trunk-max-radius-mm", type=float, default=12.0)
    radius_tune_scope = radius_tuning.add_mutually_exclusive_group()
    radius_tune_scope.add_argument("--tune-review-edges-only", dest="tune_review_edges_only", action="store_true", default=True)
    radius_tune_scope.add_argument("--tune-all-edges", dest="tune_review_edges_only", action="store_false")
    radius_tuning.add_argument("--report", default="outputs/reports/vessel_radius_tuning_stage001.md")

    domain_repair = subparsers.add_parser(
        "repair-vascular-domain-connectivity",
        help="Prune unseeded arterial/venous voxel islands and optionally reconnect graph-seeded domain components.",
    )
    domain_repair.add_argument("--voxelized-spec", required=True, help="Voxelized vascular-network spec YAML.")
    domain_repair.add_argument("--graph", default=None, help="Optional vascular graph override. Defaults to voxelized spec source graph.")
    domain_repair.add_argument("--combined-labels", default=None, help="Optional anatomy label override. Defaults to voxelized spec source labels.")
    domain_repair.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    domain_repair.add_argument("--case-id", default="vascular_domain_connectivity_repaired")
    domain_repair.add_argument("--output-dir", default="outputs/digital/vascular_domain_connectivity_repaired")
    domain_repair.add_argument("--sample-step-mm", type=float, default=None)
    domain_repair.add_argument("--seed-search-radius-voxels", type=int, default=2)
    domain_repair.add_argument("--max-unseeded-component-voxels", type=int, default=500)
    domain_repair.add_argument("--connector-radius-mm", type=float, default=0.8)
    domain_repair_connect = domain_repair.add_mutually_exclusive_group()
    domain_repair_connect.add_argument("--connect-seeded-components", dest="connect_seeded_components", action="store_true", default=True)
    domain_repair_connect.add_argument("--no-connect-seeded-components", dest="connect_seeded_components", action="store_false")
    domain_repair.add_argument("--contrast-mode", choices=["arterial", "all", "none"], default=None)
    domain_repair.add_argument("--vessel-wall-thickness-mm", type=float, default=None)
    domain_repair_volumes = domain_repair.add_mutually_exclusive_group()
    domain_repair_volumes.add_argument("--write-material-volumes", dest="write_material_volumes", action="store_true", default=True)
    domain_repair_volumes.add_argument("--skip-material-volumes", dest="write_material_volumes", action="store_false")
    domain_repair.add_argument("--report", default="outputs/reports/vascular_domain_connectivity_repair_stage001.md")

    flow_boundary = subparsers.add_parser(
        "build-flow-boundary-package",
        help="Build solver-ready boundary-condition metadata from cleaned vascular network masks.",
    )
    flow_boundary.add_argument("--voxelized-spec", required=True, help="Voxelized vascular-network spec YAML.")
    flow_boundary.add_argument("--graph", default=None, help="Optional graph YAML override. Defaults to the graph in the voxelized spec.")
    flow_boundary.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    flow_boundary.add_argument("--output-dir", default="outputs/sim/flow_boundary_conditions")
    flow_boundary.add_argument("--arterial-inlet-flow-ml-s", type=float, default=80.0)
    flow_boundary.add_argument("--nominal-outlet-pressure-drop-pa", type=float, default=8000.0)
    flow_boundary.add_argument("--venous-outlet-pressure-pa", type=float, default=667.0)
    flow_boundary.add_argument("--boundary-slab-thickness-mm", type=float, default=5.0)
    flow_boundary.add_argument("--report", default="outputs/reports/flow_boundary_conditions_stage001.md")

    flow_1d = subparsers.add_parser(
        "build-flow-1d-model",
        help="Build a first-pass steady 1D flow model from the vascular graph and boundary package.",
    )
    flow_1d.add_argument("--graph", required=True, help="Vascular network graph YAML.")
    flow_1d.add_argument("--boundary-config", required=True, help="Flow boundary-condition YAML.")
    flow_1d.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    flow_1d.add_argument("--output-dir", default="outputs/sim/flow_1d")
    flow_1d.add_argument("--blood-viscosity-cp", type=float, default=3.5)
    flow_1d.add_argument("--arterial-inlet-pressure-pa", type=float, default=13332.0)
    flow_1d.add_argument(
        "--venous-outlet-pressure-pa",
        type=float,
        default=None,
        help="Optional override. Defaults to the venous outlet pressure in the boundary package.",
    )
    flow_1d.add_argument("--report", default="outputs/reports/flow_1d_model_stage001.md")

    pulsatile_flow = subparsers.add_parser(
        "build-pulsatile-flow-model",
        help="Build a first-pass pulsatile 1D flow simulation from the steady 1D model.",
    )
    pulsatile_flow.add_argument("--flow-1d-model", required=True, help="Steady 1D flow model YAML.")
    pulsatile_flow.add_argument("--boundary-config", required=True, help="Flow boundary-condition YAML.")
    pulsatile_flow.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    pulsatile_flow.add_argument("--output-dir", default="outputs/sim/flow_pulsatile")
    pulsatile_flow.add_argument("--heart-rate-bpm", type=float, default=60.0)
    pulsatile_flow.add_argument("--samples-per-cycle", type=int, default=160)
    pulsatile_flow.add_argument("--settling-cycles", type=int, default=3)
    pulsatile_flow.add_argument("--rcr-proximal-resistance-fraction", type=float, default=0.1)
    pulsatile_flow.add_argument("--rcr-time-constant-s", type=float, default=1.2)
    pulsatile_flow.add_argument("--venous-pulsatility-fraction", type=float, default=0.35)
    pulsatile_flow.add_argument("--venous-phase-lag-fraction", type=float, default=0.15)
    pulsatile_flow.add_argument("--pressure-reference-weight", type=float, default=50.0)
    pulsatile_flow.add_argument("--report", default="outputs/reports/flow_pulsatile_model_stage001.md")

    coupled_flow = subparsers.add_parser(
        "build-coupled-pulsatile-flow-model",
        help="Build a graph-coupled pulsatile 1D flow simulation with dynamic arterial RCR outlet splits.",
    )
    coupled_flow.add_argument("--flow-1d-model", required=True, help="Steady 1D flow model YAML.")
    coupled_flow.add_argument("--boundary-config", required=True, help="Flow boundary-condition YAML.")
    coupled_flow.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    coupled_flow.add_argument("--output-dir", default="outputs/sim/flow_coupled_pulsatile")
    coupled_flow.add_argument("--heart-rate-bpm", type=float, default=60.0)
    coupled_flow.add_argument("--samples-per-cycle", type=int, default=160)
    coupled_flow.add_argument("--settling-cycles", type=int, default=3)
    coupled_flow.add_argument("--rcr-proximal-resistance-fraction", type=float, default=0.1)
    coupled_flow.add_argument("--rcr-time-constant-s", type=float, default=1.2)
    coupled_flow.add_argument("--venous-pulsatility-fraction", type=float, default=0.35)
    coupled_flow.add_argument("--venous-phase-lag-fraction", type=float, default=0.15)
    coupled_flow.add_argument("--report", default="outputs/reports/flow_coupled_pulsatile_model_stage001.md")

    flow4d = subparsers.add_parser(
        "render-4d-flow",
        help="Render time-resolved vascular flow frames from graph geometry and coupled flow time series.",
    )
    flow4d.add_argument("--graph", required=True, help="Vascular network graph YAML.")
    flow4d.add_argument("--edge-timeseries", required=True, help="Coupled edge time-series CSV.")
    flow4d.add_argument("--node-timeseries", required=True, help="Coupled node time-series CSV.")
    flow4d.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    flow4d.add_argument("--output-dir", default="outputs/sim/flow_4d_visualization")
    flow4d.add_argument("--context-scene-spec", default=None, help="Optional 3D scene spec for transparent phantom context.")
    flow4d.add_argument("--color-by", choices=["velocity", "pressure", "flow"], default="velocity")
    flow4d.add_argument("--frame-count", type=int, default=32)
    flow4d.add_argument("--view-elev", type=float, default=18.0)
    flow4d.add_argument("--view-azim", type=float, default=-58.0)
    flow4d.add_argument("--zoom", type=float, default=1.08)
    flow4d.add_argument("--context-groups", nargs="+", default=["body_envelope", "bone", "lungs", "liver", "kidneys"])
    flow4d.add_argument("--max-context-triangles-per-group", type=int, default=3500)
    flow4d.add_argument("--gif-duration-ms", type=int, default=110)
    labels = flow4d.add_mutually_exclusive_group()
    labels.add_argument("--label-boundary-nodes", dest="label_boundary_nodes", action="store_true", default=True)
    labels.add_argument("--no-label-boundary-nodes", dest="label_boundary_nodes", action="store_false")
    flow4d.add_argument("--report", default="outputs/reports/flow_4d_visualization_stage001.md")

    rt_qa = subparsers.add_parser(
        "build-radiotherapy-qa-package",
        help="Build RT-ready HU/density/RED maps, DVH masks, and PyMedPhys placeholders.",
    )
    rt_qa.add_argument("--combined-spec", required=True, help="Combined digital phantom spec YAML.")
    rt_qa.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    rt_qa.add_argument("--output-dir", default="outputs/radiotherapy/qa_package")
    rt_qa.add_argument("--scenario", choices=["blood", "contrast"], default="blood")
    rt_qa.add_argument("--target-radius-mm", type=float, default=12.0)
    rt_qa.add_argument("--ptv-margin-mm", type=float, default=5.0)
    rt_qa.add_argument("--report", default="outputs/reports/radiotherapy_qa_package_stage001.md")

    rt_plan = subparsers.add_parser(
        "build-rt-planning-bundle",
        help="Export DICOM-RT-style planning handoff files and static-vs-pulsatile dose metrics.",
    )
    rt_plan.add_argument("--rt-package-spec", required=True, help="Radiotherapy QA package spec YAML.")
    rt_plan.add_argument(
        "--coupled-flow-model",
        default=None,
        help="Optional coupled pulsatile flow model YAML used to derive vascular dose perturbation amplitude.",
    )
    rt_plan.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    rt_plan.add_argument("--output-dir", default="outputs/radiotherapy/planning_bundle")
    rt_plan.add_argument("--prescription-dose-gy", type=float, default=20.0)
    rt_plan.add_argument(
        "--vascular-dose-sensitivity",
        type=float,
        default=0.015,
        help="Maximum local dose perturbation per unit inlet-flow amplitude near vascular voxels.",
    )
    dicom = rt_plan.add_mutually_exclusive_group()
    dicom.add_argument("--export-dicom", dest="export_dicom", action="store_true", default=True)
    dicom.add_argument("--skip-dicom", dest="export_dicom", action="store_false")
    rt_plan.add_argument("--report", default="outputs/reports/rt_planning_bundle_stage001.md")

    spatial_rt_flow = subparsers.add_parser(
        "analyze-spatial-rt-flow-coupling",
        help="Rank vascular graph edges by spatial proximity to RT dose/target regions and pulsatile flow dynamics.",
    )
    spatial_rt_flow.add_argument("--rt-package-spec", required=True, help="Radiotherapy QA package spec YAML.")
    spatial_rt_flow.add_argument("--rt-planning-spec", required=True, help="RT planning bundle spec YAML.")
    spatial_rt_flow.add_argument("--vascular-graph", required=True, help="Variant vascular graph YAML.")
    spatial_rt_flow.add_argument("--edge-timeseries", required=True, help="Coupled pulsatile edge time-series CSV.")
    spatial_rt_flow.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    spatial_rt_flow.add_argument("--output-dir", default="outputs/experiments/spatial_rt_flow_coupling")
    spatial_rt_flow.add_argument("--sample-step-mm", type=float, default=2.0)
    spatial_rt_flow.add_argument("--influence-radius-mm", type=float, default=25.0)
    spatial_rt_flow.add_argument(
        "--coordinate-mode",
        choices=["voxel-mm", "nifti-affine"],
        default="voxel-mm",
        help="Coordinate transform for graph points. Use voxel-mm for project graph scaffolds.",
    )
    spatial_rt_flow.add_argument("--report", default="outputs/reports/spatial_rt_flow_coupling_stage001.md")

    spatial_rt_dose = subparsers.add_parser(
        "build-spatial-rt-flow-dose-model",
        help="Generate spatially varying pulsatile RT dose states from vascular graph edge waveforms.",
    )
    spatial_rt_dose.add_argument("--rt-package-spec", required=True, help="Radiotherapy QA package spec YAML.")
    spatial_rt_dose.add_argument("--rt-planning-spec", required=True, help="RT planning bundle spec YAML.")
    spatial_rt_dose.add_argument("--vascular-graph", required=True, help="Variant vascular graph YAML.")
    spatial_rt_dose.add_argument("--edge-timeseries", required=True, help="Coupled pulsatile edge time-series CSV.")
    spatial_rt_dose.add_argument("--edge-coupling-csv", required=True, help="Spatial RT-flow edge coupling CSV.")
    spatial_rt_dose.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    spatial_rt_dose.add_argument("--output-dir", default="outputs/radiotherapy/spatial_rt_flow_dose")
    spatial_rt_dose.add_argument("--sample-step-mm", type=float, default=2.0)
    spatial_rt_dose.add_argument("--influence-falloff-mm", type=float, default=7.5)
    spatial_rt_dose.add_argument("--vascular-dose-sensitivity", type=float, default=None)
    spatial_rt_dose.add_argument("--max-fractional-perturbation", type=float, default=0.05)
    spatial_rt_dose.add_argument("--max-edges", type=int, default=12)
    spatial_rt_dose.add_argument("--min-coupling-score", type=float, default=0.0)
    spatial_rt_dose.add_argument(
        "--coordinate-mode",
        choices=["voxel-mm", "nifti-affine"],
        default="voxel-mm",
        help="Coordinate transform for graph points. Use voxel-mm for project graph scaffolds.",
    )
    spatial_rt_dose.add_argument("--report", default="outputs/reports/spatial_rt_flow_dose_model_stage001.md")

    rt_flow_compare = subparsers.add_parser(
        "compare-scalar-vs-spatial-rt-flow-dose",
        help="Compare the older scalar RT-flow perturbation model against the spatial edge-weighted model.",
    )
    rt_flow_compare.add_argument("--scalar-rt-planning-spec", required=True, help="Scalar RT planning bundle spec YAML.")
    rt_flow_compare.add_argument("--spatial-rt-flow-dose-spec", required=True, help="Spatial RT-flow dose model spec YAML.")
    rt_flow_compare.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    rt_flow_compare.add_argument("--output-dir", default="outputs/experiments/scalar_vs_spatial_rt_flow")
    rt_flow_compare.add_argument("--report", default="outputs/reports/scalar_vs_spatial_rt_flow_dose_stage001.md")

    status_atlas = subparsers.add_parser(
        "build-current-phantom-status-atlas",
        help="Build a consolidated figure atlas and technical status report from the current digital phantom outputs.",
    )
    status_atlas.add_argument("--case-id", default="ct_org_label_population8_pca_modes_stage001_mode01_pos")
    status_atlas.add_argument("--output-dir", default="outputs/reports/status_atlas")
    status_atlas.add_argument("--report", default="outputs/reports/current_phantom_status_atlas_stage001.md")

    corrected_status = subparsers.add_parser(
        "build-corrected-branch-status-report",
        help="Build a consolidated status report for the corrected CT-grid branch-labelled vascular phantom.",
    )
    corrected_status.add_argument("--case-id", default="mode03_neg_branch_ctgrid_corrected_flow")
    corrected_status.add_argument("--output-dir", default="outputs/reports/corrected_branch_status")
    corrected_status.add_argument("--vessel-flow-manifest", default="outputs/digital/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/mode03_neg_branch_ctgrid_corrected_flow_label_vessel_flow_domain_manifest_v001.yaml")
    corrected_status.add_argument("--vessel-flow-spec", default="outputs/digital/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/mode03_neg_branch_ctgrid_corrected_flow_label_vessel_flow_domain_spec_v001.yaml")
    corrected_status.add_argument("--vessel-flow-preview", default="outputs/digital/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/mode03_neg_branch_ctgrid_corrected_flow_label_vessel_flow_domain_preview_v001.png")
    corrected_status.add_argument("--rt-package-spec", default="outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/qa_package/mode03_neg_branch_ctgrid_corrected_flow_radiotherapy_qa_package_spec_v001.yaml")
    corrected_status.add_argument("--rt-qa-preview", default="outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/qa_package/mode03_neg_branch_ctgrid_corrected_flow_radiotherapy_qa_preview_v001.png")
    corrected_status.add_argument("--coupled-flow-model", default="outputs/sim/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/flow_coupled_pulsatile/mode03_neg_branch_ctgrid_corrected_flow_coupled_pulsatile_flow_model_v001.yaml")
    corrected_status.add_argument("--coupled-flow-preview", default="outputs/sim/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/flow_coupled_pulsatile/plots/mode03_neg_branch_ctgrid_corrected_flow_coupled_pulsatile_pressure_flow_preview_v001.png")
    corrected_status.add_argument("--flow4d-spec", default="outputs/sim/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/flow_4d_visualization/mode03_neg_branch_ctgrid_corrected_flow_flow4d_visualization_spec_v001.yaml")
    corrected_status.add_argument("--spatial-coupling-spec", default="outputs/experiments/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_coupling/mode03_neg_branch_ctgrid_corrected_flow_spatial_rt_flow_coupling_spec_v001.yaml")
    corrected_status.add_argument("--spatial-coupling-csv", default="outputs/experiments/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_coupling/mode03_neg_branch_ctgrid_corrected_flow_spatial_rt_flow_edge_coupling_v001.csv")
    corrected_status.add_argument("--spatial-coupling-preview", default="outputs/experiments/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_coupling/mode03_neg_branch_ctgrid_corrected_flow_spatial_rt_flow_coupling_preview_v001.png")
    corrected_status.add_argument("--spatial-dose-spec", default="outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/mode03_neg_branch_ctgrid_corrected_flow_rt_spatial_flow_dose_model_spec_v001.yaml")
    corrected_status.add_argument("--spatial-dose-preview", default="outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/mode03_neg_branch_ctgrid_corrected_flow_rt_spatial_flow_dose_model_preview_v001.png")
    corrected_status.add_argument("--gamma-spec", default="outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/dose_gamma_qa/mode03_neg_branch_ctgrid_corrected_flow_spatial_dose_gamma_qa_spec_v001.yaml")
    corrected_status.add_argument("--gamma-qa-preview", default="outputs/radiotherapy/label_vessel_flow_domain/mode03_neg_branch_ctgrid_corrected_flow/spatial_rt_flow_dose/dose_gamma_qa/mode03_neg_branch_ctgrid_corrected_flow_spatial_dose_gamma_qa_preview_v001.png")
    corrected_status.add_argument("--report", default="outputs/reports/mode03_neg_branch_ctgrid_corrected_status_report.md")

    corrected_release = subparsers.add_parser(
        "build-corrected-branch-release-package",
        help="Build a disk-light release bundle from the corrected branch-labelled status manifest.",
    )
    corrected_release.add_argument("--status-manifest", default="outputs/reports/corrected_branch_status/mode03_neg_branch_ctgrid_corrected_flow_corrected_branch_status_manifest_v001.yaml")
    corrected_release.add_argument("--case-id", default="mode03_neg_branch_ctgrid_corrected_flow")
    corrected_release.add_argument("--release-id", default=None)
    corrected_release.add_argument("--output-dir", default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1")
    corrected_release.add_argument("--large-file-threshold-mb", type=float, default=5.0)
    corrected_release_copy = corrected_release.add_mutually_exclusive_group()
    corrected_release_copy.add_argument("--copy-small-artifacts", dest="copy_small_artifacts", action="store_true", default=True)
    corrected_release_copy.add_argument("--index-only", dest="copy_small_artifacts", action="store_false")
    corrected_release.add_argument("--report", default="outputs/reports/mode03_neg_branch_ctgrid_corrected_release_bundle.md")

    corrected_release_audit = subparsers.add_parser(
        "audit-corrected-branch-release-package",
        help="Audit the corrected branch-labelled release bundle for reproducibility, QA evidence, and clinical-claim gaps.",
    )
    corrected_release_audit.add_argument(
        "--release-manifest",
        default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/mode03_neg_branch_ctgrid_corrected_flow_rc1_release_manifest_v001.yaml",
    )
    corrected_release_audit.add_argument(
        "--output-dir",
        default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/readiness_audit",
    )
    corrected_release_audit.add_argument(
        "--audit-id",
        default="mode03_neg_branch_ctgrid_corrected_flow_rc1_readiness_audit",
    )
    corrected_release_audit.add_argument(
        "--report",
        default="outputs/reports/mode03_neg_branch_ctgrid_corrected_release_readiness_audit.md",
    )

    release_package = subparsers.add_parser(
        "build-research-release-package",
        help="Build a disk-light reproducible research release bundle for a completed digital phantom stage.",
    )
    release_package.add_argument("--case-id", default="mode03_neg_patient_build_stage007_domain_repaired")
    release_package.add_argument(
        "--stage-root",
        default="outputs/digital/patient_builds/mode03_neg_patient_build_stage007_domain_repaired",
    )
    release_package.add_argument("--reports-dir", default="outputs/reports")
    release_package.add_argument("--output-dir", default="outputs/releases/mode03_neg_stage007_rc1")
    release_package.add_argument("--release-id", default="mode03_neg_stage007_rc1")
    release_package.add_argument("--large-file-threshold-mb", type=float, default=25.0)
    release_copy = release_package.add_mutually_exclusive_group()
    release_copy.add_argument("--copy-small-artifacts", dest="copy_small_artifacts", action="store_true", default=True)
    release_copy.add_argument("--index-only", dest="copy_small_artifacts", action="store_false")
    release_package.add_argument("--report", default="outputs/reports/mode03_neg_stage007_research_release_candidate.md")

    release_audit = subparsers.add_parser(
        "audit-research-release-package",
        help="Score a research release bundle for reproducibility, research readiness, and clinical-claim gaps.",
    )
    release_audit.add_argument(
        "--release-manifest",
        default="outputs/releases/mode03_neg_stage007_rc1/mode03_neg_stage007_rc1_release_manifest_v001.yaml",
    )
    release_audit.add_argument("--output-dir", default="outputs/releases/mode03_neg_stage007_rc1/readiness_audit")
    release_audit.add_argument("--audit-id", default="mode03_neg_stage007_rc1_readiness_audit")
    release_audit.add_argument("--report", default="outputs/reports/mode03_neg_stage007_release_readiness_audit.md")

    stage007_promote = subparsers.add_parser(
        "promote-stage007-clean-baseline",
        help="Promote the clean Stage 007 BTCV release as the active research baseline and write pointer files.",
    )
    stage007_promote.add_argument(
        "--release-manifest",
        default=(
            "outputs/releases/stage007_left_iliac_radius_clean_rc1/"
            "stage007_left_iliac_radius_clean_rc1_release_manifest_v001.yaml"
        ),
    )
    stage007_promote.add_argument(
        "--stage-root",
        default="outputs/patient_case_adapter/btcv_case0001_stage007_ivc_branch_reroute",
    )
    stage007_promote.add_argument("--case-id", default="btcv_case0001_stage007_left_iliac_radius_clean")
    stage007_promote.add_argument("--baseline-id", default="stage007_left_iliac_radius_clean_rc1")
    stage007_promote.add_argument("--graph", default=None, help="Optional clean vascular graph YAML override.")
    stage007_promote.add_argument("--voxelized-spec", default=None, help="Optional clean voxelized vascular spec override.")
    stage007_promote.add_argument(
        "--release-archive",
        default="stage007_left_iliac_radius_clean_rc1_compact_release.tar.gz",
    )
    stage007_promote_alias = stage007_promote.add_mutually_exclusive_group()
    stage007_promote_alias.add_argument(
        "--write-accepted-aliases",
        dest="write_accepted_aliases",
        action="store_true",
        default=True,
        help="Also update latest_stage007_accepted_* aliases to this clean active baseline.",
    )
    stage007_promote_alias.add_argument(
        "--preserve-accepted-aliases",
        dest="write_accepted_aliases",
        action="store_false",
        help="Leave older latest_stage007_accepted_* aliases untouched.",
    )
    stage007_promote.add_argument(
        "--report",
        default="outputs/reports/btcv_case0001_stage007_left_iliac_radius_clean_active_baseline_promotion.md",
    )

    stage007_smoke = subparsers.add_parser(
        "run-stage007-acceptance-smoke",
        help="Run lightweight acceptance checks on the active clean Stage 007 release artifacts.",
    )
    stage007_smoke.add_argument(
        "--release-manifest",
        default=(
            "outputs/releases/stage007_left_iliac_radius_clean_rc1/"
            "stage007_left_iliac_radius_clean_rc1_release_manifest_v001.yaml"
        ),
    )
    stage007_smoke.add_argument("--baseline-manifest", default=None)
    stage007_smoke.add_argument(
        "--release-archive",
        default="stage007_left_iliac_radius_clean_rc1_compact_release.tar.gz",
    )
    stage007_smoke.add_argument("--case-id", default=None)
    stage007_smoke.add_argument("--baseline-id", default=None)
    stage007_smoke.add_argument(
        "--output-dir",
        default="outputs/acceptance/stage007_left_iliac_radius_clean",
    )
    stage007_smoke.add_argument("--flow-mass-residual-threshold-ml-s", type=float, default=1e-4)
    stage007_smoke.add_argument("--flow-split-review-threshold-pp", type=float, default=10.0)
    stage007_smoke.add_argument("--flow-split-fail-threshold-pp", type=float, default=15.0)
    stage007_smoke.add_argument("--min-boundary-count", type=int, default=10)
    stage007_smoke.add_argument(
        "--report",
        default="outputs/reports/btcv_case0001_stage007_left_iliac_radius_clean_acceptance_smoke.md",
    )

    validation_roadmap = subparsers.add_parser(
        "build-validation-roadmap",
        help="Convert a release-readiness audit into a clinical-validation gap-closure protocol and task plan.",
    )
    validation_roadmap.add_argument(
        "--readiness-audit",
        default="outputs/releases/mode03_neg_stage007_rc1/readiness_audit/mode03_neg_stage007_rc1_readiness_audit_audit_v001.yaml",
    )
    validation_roadmap.add_argument("--roadmap-csv", default=None)
    validation_roadmap.add_argument("--output-dir", default="outputs/releases/mode03_neg_stage007_rc1/validation_roadmap")
    validation_roadmap.add_argument("--roadmap-id", default="mode03_neg_stage007_validation_roadmap")
    validation_roadmap.add_argument("--report", default="outputs/reports/mode03_neg_stage007_validation_roadmap.md")

    demonstrator = subparsers.add_parser(
        "build-research-demonstrator-package",
        help="Assemble a publishable engineering/research demonstrator package from the corrected branch release outputs.",
    )
    demonstrator.add_argument(
        "--release-manifest",
        default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/mode03_neg_branch_ctgrid_corrected_flow_rc1_release_manifest_v001.yaml",
    )
    demonstrator.add_argument(
        "--audit-yaml",
        default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/readiness_audit/mode03_neg_branch_ctgrid_corrected_flow_rc1_readiness_audit_audit_v001.yaml",
    )
    demonstrator.add_argument(
        "--status-manifest",
        default="outputs/reports/corrected_branch_status/mode03_neg_branch_ctgrid_corrected_flow_corrected_branch_status_manifest_v001.yaml",
    )
    demonstrator.add_argument(
        "--validation-roadmap",
        default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/validation_roadmap/mode03_neg_branch_ctgrid_corrected_flow_rc1_validation_roadmap_roadmap_v001.yaml",
    )
    demonstrator.add_argument(
        "--validation-intake",
        default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/validation_intake_medseg_harmonized_partial_p1/medseg_abdominal_vasculature_case001_harmonized_partial_p1_intake_manifest_v001.yaml",
    )
    demonstrator.add_argument(
        "--vessel-harmonization",
        default="outputs/digital/vessel_label_harmonization/medseg_abdominal_vasculature_case001_partial_p1/medseg_abdominal_vasculature_case001_partial_p1_vessel_label_harmonization_manifest_v001.yaml",
    )
    demonstrator.add_argument(
        "--output-dir",
        default="outputs/releases/mode03_neg_branch_ctgrid_corrected_rc1/research_demonstrator",
    )
    demonstrator.add_argument("--package-id", default=None)
    demonstrator.add_argument(
        "--report",
        default="outputs/reports/mode03_neg_branch_ctgrid_corrected_research_demonstrator.md",
    )

    validation_intake = subparsers.add_parser(
        "build-validation-intake-package",
        help="Create and score a P1 CT/CTA/CTV validation case-intake package for patient-specific vascular grounding.",
    )
    validation_intake.add_argument("--cases-csv", default=None, help="Candidate case CSV. If omitted, only a blank template is generated.")
    validation_intake.add_argument("--output-dir", default="outputs/releases/mode03_neg_stage007_rc1/validation_intake")
    validation_intake.add_argument("--intake-id", default="mode03_neg_stage007_p1_cta_ctv_intake")
    validation_intake.add_argument(
        "--required-vessel-labels",
        nargs="*",
        type=int,
        default=None,
        help="Required vessel label IDs. Defaults to the P1 abdominal arterial/venous branch set.",
    )
    validation_intake.add_argument("--report", default="outputs/reports/mode03_neg_stage007_p1_validation_intake.md")

    validation_discovery = subparsers.add_parser(
        "discover-validation-candidates",
        help="Scan local project data folders and create a P1 CT/CTA/CTV candidate CSV for intake scoring.",
    )
    validation_discovery.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Folder or manifest to scan. Can be repeated. Defaults to staged data plus patient-input adapter outputs.",
    )
    validation_discovery.add_argument("--output-dir", default="outputs/releases/mode03_neg_stage007_rc1/validation_discovery")
    validation_discovery.add_argument("--discovery-id", default="mode03_neg_stage007_p1_candidate_discovery")
    validation_discovery.add_argument(
        "--required-vessel-labels",
        nargs="*",
        type=int,
        default=None,
        help="Required vessel label IDs. Defaults to the P1 abdominal arterial/venous branch set.",
    )
    validation_discovery.add_argument("--max-ct-org-cases", type=int, default=10)
    validation_discovery.add_argument("--max-loose-nifti-cases", type=int, default=20)
    validation_discovery.add_argument("--report", default="outputs/reports/mode03_neg_stage007_p1_candidate_discovery.md")

    validation_case = subparsers.add_parser(
        "stage-validation-case",
        help="Create a standardized P1 validation-case manifest and intake CSV from CT/CTA/CTV and segmentation paths.",
    )
    validation_case.add_argument("--case-id", required=True)
    validation_case.add_argument("--source-dataset", required=True)
    validation_case.add_argument("--ct", default=None, help="Primary CT NIfTI or DICOM directory.")
    validation_case.add_argument("--cta", default=None, help="CTA NIfTI or DICOM directory.")
    validation_case.add_argument("--ctv", default=None, help="CTV NIfTI or DICOM directory.")
    validation_case.add_argument("--organ-seg", default=None, help="Organ/material segmentation on the CT grid.")
    validation_case.add_argument("--vessel-seg", default=None, help="Branch-labelled vessel segmentation.")
    validation_case.add_argument("--vessel-label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    validation_case.add_argument("--output-dir", default="data/validation/p1_cases")
    validation_case.add_argument(
        "--required-vessel-labels",
        nargs="*",
        type=int,
        default=None,
        help="Required vessel label IDs. Defaults to the P1 abdominal arterial/venous branch set.",
    )
    validation_case.add_argument("--access-status", default="local_review_required")
    validation_case.add_argument("--notes", default="")
    validation_case.add_argument("--copy-inputs", action="store_true", help="Copy input files into the staged case folder. Default is reference-only.")
    validation_case.add_argument("--report", default=None)

    vessel_labels = subparsers.add_parser(
        "harmonize-vessel-labels",
        help="Generate or apply a source-to-P1 vessel label mapping for CTA/CTV segmentations.",
    )
    vessel_labels.add_argument("--vessel-seg", required=True, help="Source labelled vessel segmentation NIfTI.")
    vessel_labels.add_argument("--case-id", required=True)
    vessel_labels.add_argument("--output-dir", default="outputs/digital/vessel_label_harmonization")
    vessel_labels.add_argument("--target-label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    vessel_labels.add_argument("--mapping-csv", default=None, help="CSV with source_label and target_label columns. If omitted, only a template is written unless --auto-identity is set.")
    vessel_labels.add_argument(
        "--required-vessel-labels",
        nargs="*",
        type=int,
        default=None,
        help="Required vessel label IDs. Defaults to the P1 abdominal arterial/venous branch set.",
    )
    vessel_labels.add_argument("--auto-identity", action="store_true", help="Map source labels to identical target labels when those IDs exist in the target config.")
    vessel_labels.add_argument("--unmapped-policy", choices=["zero", "preserve"], default="zero")
    vessel_labels.add_argument("--report", default=None)

    vessel_promotion = subparsers.add_parser(
        "promote-harmonized-vessel-case",
        help="Create a promoted staged validation case that replaces its vessel mask with a harmonized vessel-label output.",
    )
    vessel_promotion.add_argument("--staged-case-manifest", required=True)
    vessel_promotion.add_argument("--vessel-harmonization-manifest", required=True)
    vessel_promotion.add_argument("--promoted-case-id", default=None)
    vessel_promotion.add_argument("--output-dir", default="data/validation/p1_cases")
    vessel_promotion.add_argument("--report", default=None)

    vessel_ct_grid = subparsers.add_parser(
        "resample-vessel-to-ct-grid",
        help="Nearest-neighbor resample a labelled vessel mask onto a staged CT grid with geometry QA.",
    )
    vessel_ct_grid.add_argument("--ct", default=None, help="Target CT NIfTI. Optional if --staged-case-manifest is supplied.")
    vessel_ct_grid.add_argument("--vessel-seg", default=None, help="Source labelled vessel NIfTI. Optional if --staged-case-manifest is supplied.")
    vessel_ct_grid.add_argument("--staged-case-manifest", default=None, help="Staged validation case manifest containing CT, organ, and vessel paths.")
    vessel_ct_grid.add_argument("--target-mask", default=None, help="Optional target organ/body mask for centered-bbox template placement.")
    vessel_ct_grid.add_argument("--case-id", default="vessel_ct_grid_resample")
    vessel_ct_grid.add_argument("--output-dir", default="outputs/digital/vessel_ct_grid_resample")
    vessel_ct_grid.add_argument("--alignment-mode", choices=["header-affine", "centered-bbox"], default="header-affine")
    vessel_ct_grid.add_argument(
        "--required-vessel-labels",
        nargs="*",
        type=int,
        default=None,
        help="Required vessel label IDs. Defaults to the P1 abdominal arterial/venous branch set.",
    )
    vessel_ct_grid.add_argument("--report", default=None)

    vessel_label_qa = subparsers.add_parser(
        "qa-vessel-label-anatomy",
        help="Run organ-aware QA on a CT-grid labelled vessel mask against material/anatomy labels.",
    )
    vessel_label_qa.add_argument("--anatomy-labels", required=True, help="Material/anatomy label NIfTI on the CT grid.")
    vessel_label_qa.add_argument("--vessel-labels", required=True, help="Labelled vessel NIfTI on the same grid.")
    vessel_label_qa.add_argument("--case-id", default="vessel_label_anatomy_qa")
    vessel_label_qa.add_argument("--output-dir", default="outputs/validation/vessel_label_anatomy_qa")
    vessel_label_qa.add_argument("--vessel-label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    vessel_label_qa.add_argument(
        "--required-vessel-labels",
        nargs="*",
        type=int,
        default=None,
        help="Required vessel label IDs. Defaults to the P1 abdominal arterial/venous branch set.",
    )
    vessel_label_qa.add_argument("--report", default=None)

    vessel_label_correction = subparsers.add_parser(
        "correct-vessel-label-anatomy",
        help="Clear labelled vessel voxels from bone/outside-body regions and regrow labels into nearby allowed anatomy.",
    )
    vessel_label_correction.add_argument("--anatomy-labels", required=True, help="Material/anatomy label NIfTI on the CT grid.")
    vessel_label_correction.add_argument("--vessel-labels", required=True, help="Labelled vessel NIfTI on the same grid.")
    vessel_label_correction.add_argument("--case-id", default="vessel_label_anatomy_correction")
    vessel_label_correction.add_argument("--output-dir", default="outputs/digital/vessel_label_anatomy_correction")
    vessel_label_correction.add_argument("--vessel-label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    vessel_label_correction.add_argument("--max-regrow-iterations", type=int, default=8)
    vessel_label_correction.add_argument("--report", default=None)

    label_vessel_flow_domain = subparsers.add_parser(
        "build-label-vessel-flow-domain",
        help="Convert a corrected CT-grid labelled vessel mask into solver-ready arterial/venous flow-domain volumes.",
    )
    label_vessel_flow_domain.add_argument("--anatomy-labels", required=True, help="Material/anatomy label NIfTI on the CT grid.")
    label_vessel_flow_domain.add_argument("--vessel-labels", required=True, help="Corrected labelled vessel NIfTI on the same CT grid.")
    label_vessel_flow_domain.add_argument("--graph", required=True, help="Vascular graph YAML used for inlet/outlet topology.")
    label_vessel_flow_domain.add_argument("--case-id", default="label_vessel_flow_domain")
    label_vessel_flow_domain.add_argument("--output-dir", default="outputs/digital/label_vessel_flow_domain")
    label_vessel_flow_domain.add_argument("--vessel-label-config", default="configs/labelmaps/medseg_abdominal_vasculature.yaml")
    label_vessel_flow_domain.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    label_vessel_flow_domain.add_argument("--vessel-wall-thickness-mm", type=float, default=1.5)
    label_vessel_flow_domain.add_argument("--contrast-mode", choices=["arterial", "all", "none"], default="arterial")
    label_vessel_flow_domain.add_argument("--boundary-snap-radius-mm", type=float, default=40.0)
    label_vessel_flow_snap = label_vessel_flow_domain.add_mutually_exclusive_group()
    label_vessel_flow_snap.add_argument("--snap-boundary-nodes", dest="snap_boundary_nodes", action="store_true", default=True)
    label_vessel_flow_snap.add_argument("--no-snap-boundary-nodes", dest="snap_boundary_nodes", action="store_false")
    label_vessel_flow_volumes = label_vessel_flow_domain.add_mutually_exclusive_group()
    label_vessel_flow_volumes.add_argument("--write-material-volumes", dest="write_material_volumes", action="store_true", default=False)
    label_vessel_flow_volumes.add_argument("--skip-material-volumes", dest="write_material_volumes", action="store_false")
    label_vessel_flow_domain.add_argument("--report", default=None)

    dose_gamma = subparsers.add_parser(
        "build-dose-gamma-qa",
        help="Run PyMedPhys gamma QA comparing static reference dose against evaluated dose states.",
    )
    dose_gamma.add_argument("--pymedphys-eval-config", required=True, help="PyMedPhys evaluation config YAML.")
    dose_gamma.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    dose_gamma.add_argument("--output-dir", default="outputs/radiotherapy/dose_gamma_qa")
    dose_gamma.add_argument("--dose-percent-threshold", type=float, default=None)
    dose_gamma.add_argument("--distance-mm-threshold", type=float, default=None)
    dose_gamma.add_argument("--lower-percent-dose-cutoff", type=float, default=None)
    dose_gamma.add_argument("--interp-fraction", type=float, default=3.0)
    dose_gamma.add_argument("--max-gamma", type=float, default=2.0)
    dose_gamma.add_argument("--random-subset", type=int, default=25000)
    dose_gamma.add_argument("--random-seed", type=int, default=20260526)
    volume_outputs = dose_gamma.add_mutually_exclusive_group()
    volume_outputs.add_argument("--write-volume-outputs", dest="write_volume_outputs", action="store_true", default=True)
    volume_outputs.add_argument("--skip-volume-outputs", dest="write_volume_outputs", action="store_false")
    gamma_mode = dose_gamma.add_mutually_exclusive_group()
    gamma_mode.add_argument("--local-gamma", dest="local_gamma", action="store_true", default=None)
    gamma_mode.add_argument("--global-gamma", dest="local_gamma", action="store_false")
    dose_gamma.add_argument("--report", default="outputs/reports/dose_gamma_qa_stage001.md")

    cartridge = subparsers.add_parser(
        "build-printable-cartridge",
        help="Build a watertight printable vascular flow cartridge by voxel boolean modeling.",
    )
    cartridge.add_argument("--labels", required=True, help="Path to vascular label NIfTI file.")
    cartridge.add_argument("--label-id", type=int, default=1, help="Label ID to use as the fluid lumen.")
    cartridge.add_argument("--flow-loop-spec", required=True, help="Path to flow-loop adapter spec YAML.")
    cartridge.add_argument("--centerline-csv", default=None, help="Optional centerline CSV for traceability.")
    cartridge.add_argument("--output-dir", default="outputs/cad/vascular_cartridge")
    cartridge.add_argument("--formats", nargs="+", default=["stl", "ply"])
    cartridge.add_argument("--voxel-size-mm", type=float, default=1.0)
    cartridge.add_argument("--wall-thickness-mm", type=float, default=2.5)
    cartridge.add_argument("--bore-clearance-mm", type=float, default=0.3)
    cartridge.add_argument("--pressure-tap-wall-mm", type=float, default=2.0)
    cartridge.add_argument("--target-max-faces", type=int, default=250_000)
    cartridge.add_argument("--report", default="outputs/reports/vascular_printable_cartridge_stage001.md")
    cartridge.add_argument("--qa-report", default="outputs/reports/vascular_printable_cartridge_mesh_qa_stage001.md")

    torso = subparsers.add_parser(
        "build-digital-torso",
        help="Build a CT-derived digital torso material, density, RED, and synthetic-HU volume package.",
    )
    torso.add_argument("--ct", required=True, help="Path to source CT NIfTI volume.")
    torso.add_argument("--labels", required=True, help="Path to CT-ORG label NIfTI volume.")
    torso.add_argument("--labelmap", default=str(DEFAULT_CT_ORG_LABELMAP))
    torso.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    torso.add_argument("--case-id", default="ct_org_case0")
    torso.add_argument("--output-dir", default="outputs/digital/torso")
    torso.add_argument("--body-threshold-hu", type=float, default=-500.0)
    torso.add_argument("--report", default="outputs/reports/digital_torso_stage001.md")

    combined = subparsers.add_parser(
        "build-combined-digital-phantom",
        help="Embed a vascular module into the digital torso material maps and export flow metadata.",
    )
    combined.add_argument("--torso-material-labels", required=True, help="Baseline torso material-label NIfTI.")
    combined.add_argument("--torso-body-mask", required=True, help="Baseline torso body-mask NIfTI.")
    combined.add_argument("--source-ct", required=True, help="Source CT NIfTI used for preview background.")
    combined.add_argument("--vascular-labels", required=True, help="ImageTBAD vascular label NIfTI.")
    combined.add_argument("--vascular-label-id", type=int, default=1, help="Vascular label ID to embed as fluid.")
    combined.add_argument("--flow-loop-spec", required=True, help="Flow-loop adapter/boundary YAML.")
    combined.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    combined.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    combined.add_argument("--output-dir", default="outputs/digital/combined")
    combined.add_argument("--target-center-mm", nargs=3, type=float, default=None)
    combined.add_argument("--vascular-scale", type=float, default=1.0)
    combined.add_argument("--vessel-wall-thickness-mm", type=float, default=2.0)
    combined.add_argument("--report", default="outputs/reports/combined_digital_phantom_stage001.md")

    anthropometry = subparsers.add_parser(
        "build-anthropometric-torso-morph",
        help="Create a BMI/waist/height-adapted torso phantom volume from a combined digital phantom spec.",
    )
    anthropometry.add_argument("--combined-spec", required=True, help="Combined digital phantom spec YAML.")
    anthropometry.add_argument("--case-id", default="ct_org_case0_imagetbad_case125_bmi32_waist110")
    anthropometry.add_argument("--output-dir", default="outputs/digital/anthropometric_morph")
    anthropometry.add_argument("--target-height-cm", type=float, default=175.0)
    anthropometry.add_argument("--target-weight-kg", type=float, default=None)
    anthropometry.add_argument("--target-bmi", type=float, default=32.0)
    anthropometry.add_argument("--target-waist-cm", type=float, default=110.0)
    anthropometry.add_argument("--baseline-height-cm", type=float, default=170.0)
    anthropometry.add_argument("--baseline-bmi", type=float, default=24.0)
    anthropometry.add_argument("--baseline-waist-cm", type=float, default=None)
    anthropometry.add_argument("--abdomen-center-fraction", type=float, default=0.46)
    anthropometry.add_argument("--abdomen-width-fraction", type=float, default=0.24)
    anthropometry.add_argument("--morph-mode", choices=["standard", "high-bmi"], default="standard")
    anthropometry.add_argument("--xy-padding-voxels", type=int, default=0)
    anthropometry.add_argument("--report", default="outputs/reports/anthropometric_torso_morph_stage001.md")

    statistical_anatomy = subparsers.add_parser(
        "build-statistical-anatomy-morph",
        help="Build a registration/PCA-based anatomy morph from staged population material-label segmentations.",
    )
    statistical_anatomy.add_argument("--combined-spec", required=True, help="Reference combined digital phantom spec YAML.")
    statistical_anatomy.add_argument(
        "--population-labels",
        nargs="+",
        required=True,
        help="Population material-label NIfTI files registered into the statistical shape model.",
    )
    statistical_anatomy.add_argument(
        "--population-case-ids",
        nargs="+",
        default=None,
        help="Optional IDs matching --population-labels. Defaults to file stems.",
    )
    statistical_anatomy.add_argument("--case-id", default="ct_org_population_statistical_morph")
    statistical_anatomy.add_argument("--output-dir", default="outputs/digital/statistical_anatomy_morph")
    statistical_anatomy.add_argument("--target-height-cm", type=float, default=None)
    statistical_anatomy.add_argument("--target-weight-kg", type=float, default=None)
    statistical_anatomy.add_argument("--target-bmi", type=float, default=None)
    statistical_anatomy.add_argument("--target-waist-cm", type=float, default=None)
    statistical_anatomy.add_argument("--baseline-height-cm", type=float, default=170.0)
    statistical_anatomy.add_argument("--baseline-bmi", type=float, default=24.0)
    statistical_anatomy.add_argument(
        "--mode-weights",
        nargs="*",
        type=float,
        default=[],
        help="Optional PCA mode weights in normalized feature space.",
    )
    statistical_anatomy.add_argument("--max-modes", type=int, default=3)
    statistical_anatomy.add_argument("--adipose-layer-mm", type=float, default=18.0)
    statistical_anatomy.add_argument("--report", default="outputs/reports/statistical_anatomy_morph_stage001.md")

    population_cohort = subparsers.add_parser(
        "build-population-cohort",
        help="Register staged population material-label NIfTIs and export cohort QA plus PCA-ready inputs.",
    )
    population_cohort.add_argument("--combined-spec", required=True, help="Reference combined digital phantom spec YAML.")
    population_cohort.add_argument(
        "--population-labels",
        nargs="+",
        required=True,
        help="Population material-label NIfTI files to register and QA.",
    )
    population_cohort.add_argument(
        "--population-case-ids",
        nargs="+",
        default=None,
        help="Optional IDs matching --population-labels. Defaults to file stems.",
    )
    population_cohort.add_argument("--cohort-id", default="ct_org_population_cohort")
    population_cohort.add_argument("--output-dir", default="outputs/digital/population_cohort")
    population_cohort.add_argument("--max-modes", type=int, default=6)
    population_cohort.add_argument("--min-body-dice", type=float, default=0.55)
    population_cohort.add_argument("--min-body-overlap", type=float, default=0.60)
    population_cohort.add_argument("--max-atlas-cases", type=int, default=24)
    population_cohort.add_argument("--essential-groups", nargs="+", default=["lungs", "liver", "kidneys", "bone"])
    population_cohort.add_argument("--report", default="outputs/reports/population_cohort_stage001.md")

    ct_org_stage = subparsers.add_parser(
        "stage-ct-org-label-cohort",
        help="Download CT-ORG segmentation labels and materialize disk-light population material-label NIfTIs.",
    )
    ct_org_stage.add_argument("--case-indices", nargs="+", type=int, default=list(range(10)))
    ct_org_stage.add_argument("--raw-label-dir", default="data/raw/ct_org/labels")
    ct_org_stage.add_argument("--output-dir", default="data/processed/ct_org_label_population")
    ct_org_stage.add_argument("--case-id-prefix", default="ct_org_case")
    ct_org_stage.add_argument(
        "--label-base-url",
        default="https://huggingface.co/datasets/Angelou0516/ct-org/resolve/main/labels",
    )
    ct_org_stage.add_argument("--body-padding-mm", type=float, default=35.0)
    ct_org_stage.add_argument("--adipose-layer-mm", type=float, default=18.0)
    ct_org_stage.add_argument("--force-download", action="store_true")
    ct_org_stage.add_argument("--report", default="outputs/reports/ct_org_label_population_stage001.md")

    avt_kits_stage = subparsers.add_parser(
        "stage-avt-kits-aorta-cohort",
        help="Stage AVT KiTS.zip CT/aorta NRRD cases as NIfTI files for aorta registration practice.",
    )
    avt_kits_stage.add_argument("--zip", dest="zip_path", required=True, help="Path to KiTS.zip from the AVT download.")
    avt_kits_stage.add_argument("--dataset-id", default="avt_kits_aorta_stage001")
    avt_kits_stage.add_argument("--output-dir", default="data/processed/avt_kits_aorta")
    avt_kits_stage.add_argument("--case-ids", nargs="+", default=None, help="Optional source case IDs, e.g. K1 K2.")
    avt_kits_stage.add_argument("--max-cases", type=int, default=None)
    avt_kits_stage.add_argument("--case-id-prefix", default="avt_kits")
    avt_kits_stage.add_argument("--report", default="outputs/reports/avt_kits_aorta_stage001.md")

    reg_training_stage = subparsers.add_parser(
        "stage-reg-training-testing",
        help="Stage a disk-safe subset of AVT Reg-Training-Testing.zip image/label registration pairs.",
    )
    reg_training_stage.add_argument("--zip", dest="zip_path", required=True, help="Path to Reg-Training-Testing.zip.")
    reg_training_stage.add_argument("--dataset-id", default="reg_training_testing_stage001")
    reg_training_stage.add_argument("--output-dir", default="data/processed/reg_training_testing")
    reg_training_stage.add_argument("--target-case-ids", nargs="+", default=[])
    reg_training_stage.add_argument("--max-targets", type=int, default=3)
    reg_training_stage.add_argument("--max-pairs-per-target", type=int, default=None)
    reg_training_stage.add_argument("--labels-only", dest="extract_images", action="store_false", default=True)
    reg_training_stage.add_argument("--report", default="outputs/reports/reg_training_testing_stage001.md")

    reg_training_benchmark = subparsers.add_parser(
        "build-reg-training-testing-benchmark",
        help="Benchmark propagated-label registration consistency from a staged Reg-Training-Testing subset.",
    )
    reg_training_benchmark.add_argument(
        "--staged-manifest",
        default="data/processed/reg_training_testing/reg_training_testing_stage001_manifest_v001.yaml",
    )
    reg_training_benchmark.add_argument("--dataset-id", default="reg_training_testing_benchmark_stage001")
    reg_training_benchmark.add_argument("--output-dir", default="outputs/digital/reg_training_testing_benchmark")
    reg_training_benchmark.add_argument("--labelmap", default="configs/labelmaps/btcv_abdomen.yaml")
    reg_training_benchmark.add_argument("--min-consensus-fraction", type=float, default=0.35)
    reg_training_benchmark.add_argument("--min-volume-weighted-dice", type=float, default=0.65)
    reg_training_benchmark.add_argument("--min-mean-dice", type=float, default=0.50)
    reg_training_benchmark.add_argument("--max-centroid-dispersion-mm", type=float, default=45.0)
    reg_training_benchmark.add_argument("--min-mean-ncc", type=float, default=0.15)
    reg_training_benchmark.add_argument("--intensity-sample-stride", type=int, default=8)
    reg_training_benchmark.add_argument("--report", default="outputs/reports/reg_training_testing_benchmark_stage001.md")

    reg_anchor_qa = subparsers.add_parser(
        "rank-registration-anchors",
        help="Rank registration benchmark labels into approved/review/rejected anatomy deformation anchors.",
    )
    reg_anchor_qa.add_argument(
        "--benchmark-spec",
        default="outputs/digital/reg_training_testing_benchmark/reg_training_testing_benchmark_stage001/reg_training_testing_benchmark_stage001_benchmark_spec_v001.yaml",
    )
    reg_anchor_qa.add_argument("--case-id", default="reg_training_testing_anchor_qa_stage001")
    reg_anchor_qa.add_argument("--output-dir", default="outputs/digital/registration_anchor_qa")
    reg_anchor_qa.add_argument("--approve-min-mean-dice", type=float, default=0.70)
    reg_anchor_qa.add_argument("--approve-min-target-mean-dice", type=float, default=0.70)
    reg_anchor_qa.add_argument("--approve-max-mean-volume-cv", type=float, default=0.50)
    reg_anchor_qa.add_argument("--approve-max-target-centroid-dispersion-mm", type=float, default=125.0)
    reg_anchor_qa.add_argument("--review-min-mean-dice", type=float, default=0.35)
    reg_anchor_qa.add_argument("--review-min-target-mean-dice", type=float, default=0.25)
    reg_anchor_qa.add_argument("--review-max-mean-volume-cv", type=float, default=0.85)
    reg_anchor_qa.add_argument("--review-max-target-centroid-dispersion-mm", type=float, default=185.0)
    reg_anchor_qa.add_argument("--report", default="outputs/reports/registration_anchor_qa_stage001.md")

    deformation_match = subparsers.add_parser(
        "run-deformation-match-experiment",
        help="Score BMI/height deformation variants against all currently staged scan-derived targets.",
    )
    deformation_match.add_argument("--experiment-id", default="all_available_deformation_match_stage001")
    deformation_match.add_argument("--output-dir", default="outputs/experiments/deformation_match")
    deformation_match.add_argument(
        "--ct-org-manifest",
        default="data/processed/ct_org_label_population/ct_org_label_population_manifest_v001.csv",
    )
    deformation_match.add_argument("--btcv-label", default="data/raw/btcv_abdomen/case0001/label0001.nii.gz")
    deformation_match.add_argument(
        "--reg-training-manifest",
        default="data/processed/reg_training_testing_all_labels/reg_training_testing_all_labels_stage001_manifest_v001.yaml",
    )
    deformation_match.add_argument(
        "--avt-aorta-manifest",
        default="data/processed/avt_kits_aorta/avt_kits_aorta_stage001_manifest_v001.csv",
    )
    deformation_match.add_argument(
        "--profile-spec-glob",
        default="outputs/experiments/profile_sweep/**/anthropometry/*_anthro_morph_spec_v001.yaml",
    )
    deformation_match_grid = deformation_match.add_mutually_exclusive_group()
    deformation_match_grid.add_argument("--include-height-grid", dest="include_height_grid", action="store_true", default=True)
    deformation_match_grid.add_argument("--no-height-grid", dest="include_height_grid", action="store_false")
    deformation_match.add_argument("--report", default="outputs/reports/deformation_match_experiment_stage001.md")

    pca_variants = subparsers.add_parser(
        "generate-pca-mode-variants",
        help="Generate mean and +/- PCA anatomy mode variants from a population cohort spec.",
    )
    pca_variants.add_argument("--combined-spec", required=True, help="Reference combined digital phantom spec YAML.")
    pca_variants.add_argument("--cohort-spec", required=True, help="Population cohort spec YAML from build-population-cohort.")
    pca_variants.add_argument("--case-id", default="ct_org_label_population8_pca_modes")
    pca_variants.add_argument("--output-dir", default="outputs/digital/pca_mode_variants")
    pca_variants.add_argument("--mode-count", type=int, default=3)
    pca_variants.add_argument("--amplitude", type=float, default=1.0)
    pca_variants.add_argument("--target-height-cm", type=float, default=None)
    pca_variants.add_argument("--target-weight-kg", type=float, default=None)
    pca_variants.add_argument("--target-bmi", type=float, default=None)
    pca_variants.add_argument("--target-waist-cm", type=float, default=None)
    pca_variants.add_argument("--baseline-height-cm", type=float, default=170.0)
    pca_variants.add_argument("--baseline-bmi", type=float, default=24.0)
    pca_variants.add_argument("--max-modes", type=int, default=6)
    pca_variants.add_argument("--adipose-layer-mm", type=float, default=18.0)
    pca_variants.add_argument("--report", default="outputs/reports/pca_mode_variant_atlas_stage001.md")

    pca_qa = subparsers.add_parser(
        "qa-pca-modes",
        help="Rank PCA anatomy modes and approve/reject them from variant metrics and atlas metadata.",
    )
    pca_qa.add_argument("--metrics-csv", required=True, help="PCA mode variant metrics CSV.")
    pca_qa.add_argument("--atlas-spec", required=True, help="PCA mode variant atlas spec YAML.")
    pca_qa.add_argument("--case-id", default=None)
    pca_qa.add_argument("--output-dir", default="outputs/digital/pca_mode_qa")
    pca_qa.add_argument("--max-waist-delta-cm", type=float, default=2.0)
    pca_qa.add_argument("--max-body-delta-l", type=float, default=1.25)
    pca_qa.add_argument("--max-group-delta-percent", type=float, default=35.0)
    pca_qa.add_argument("--max-vascular-volume-delta-percent", type=float, default=10.0)
    pca_qa.add_argument("--expected-vascular-components", type=int, default=1)
    pca_qa.add_argument("--min-score-for-approval", type=float, default=70.0)
    pca_qa.add_argument("--report", default="outputs/reports/pca_mode_qa_stage001.md")

    pca_set = subparsers.add_parser(
        "build-approved-pca-phantom-set",
        help="Assemble a disk-light release set containing only QA-approved PCA phantom variants.",
    )
    pca_set.add_argument("--qa-decisions", required=True, help="PCA mode QA decisions YAML.")
    pca_set.add_argument("--atlas-spec", required=True, help="PCA mode variant atlas spec YAML.")
    pca_set.add_argument("--metrics-csv", default=None, help="Optional source metrics CSV override.")
    pca_set.add_argument("--case-id", default=None)
    pca_set.add_argument("--output-dir", default="outputs/digital/approved_pca_phantom_set")
    pca_set.add_argument("--report", default="outputs/reports/approved_pca_phantom_set_stage001.md")
    pca_set.add_argument("--max-preview-columns", type=int, default=3)

    profile_adapter = subparsers.add_parser(
        "build-user-profile-adapter",
        help="Select and score approved PCA anatomies against a target BMI/waist/height profile.",
    )
    profile_adapter.add_argument("--approved-set-manifest", required=True, help="Approved PCA phantom-set manifest YAML.")
    profile_adapter.add_argument("--metrics-csv", default=None, help="Optional approved/PCA metrics CSV override.")
    profile_adapter.add_argument("--profile-id", default="demo_bmi32_waist110")
    profile_adapter.add_argument("--case-id", default=None)
    profile_adapter.add_argument("--output-dir", default="outputs/digital/user_profile_adapter")
    profile_adapter.add_argument("--target-height-cm", type=float, default=175.0)
    profile_adapter.add_argument("--target-weight-kg", type=float, default=None)
    profile_adapter.add_argument("--target-bmi", type=float, default=32.0)
    profile_adapter.add_argument("--target-waist-cm", type=float, default=110.0)
    profile_adapter.add_argument("--baseline-height-cm", type=float, default=170.0)
    profile_adapter.add_argument("--baseline-bmi", type=float, default=24.0)
    profile_adapter.add_argument("--waist-tolerance-cm", type=float, default=3.0)
    profile_adapter.add_argument("--body-volume-tolerance-l", type=float, default=1.5)
    profile_adapter.add_argument("--warning-penalty", type=float, default=8.0)
    profile_adapter.add_argument("--report", default="outputs/reports/user_profile_adapter_stage001.md")

    patient_adapter = subparsers.add_parser(
        "build-patient-phantom-adapter",
        aliases=["build-patient-phantom"],
        help="Stage and QA patient CT/CTA/CTV plus organ/vessel segmentations for phantom adaptation.",
    )
    patient_adapter.add_argument("--input-ct", default=None, help="Primary CT NIfTI or DICOM directory.")
    patient_adapter.add_argument("--input-cta", default=None, help="Optional CTA NIfTI or DICOM directory.")
    patient_adapter.add_argument("--input-ctv", default=None, help="Optional CTV NIfTI or DICOM directory.")
    patient_adapter.add_argument("--organ-seg", default=None, help="Optional organ/material segmentation NIfTI.")
    patient_adapter.add_argument("--gi-seg", default=None, help="Optional co-registered GI/bowel/colon/small-intestine segmentation NIfTI or mask directory.")
    patient_adapter.add_argument("--vessel-seg", default=None, help="Optional arterial/venous vessel segmentation NIfTI.")
    patient_adapter.add_argument("--patient-id", default="patient_demo")
    patient_adapter.add_argument("--case-id", default=None)
    patient_adapter.add_argument(
        "--adaptation-mode",
        choices=["intake-only", "ct-registered", "vascular-registered", "hybrid"],
        default="hybrid",
    )
    patient_adapter.add_argument("--output-dir", default="outputs/digital/patient_input_adapter")
    patient_adapter.add_argument("--organ-labelmap", default=str(DEFAULT_CT_ORG_LABELMAP))
    patient_adapter.add_argument("--gi-labelmap", default="configs/labelmaps/gi_tract.yaml")
    patient_adapter.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    patient_adapter.add_argument("--approved-set-manifest", default=None)
    patient_adapter.add_argument("--baseline-graph", default=None)
    patient_adapter.add_argument("--baseline-combined-spec", default=None)
    patient_adapter.add_argument("--target-height-cm", type=float, default=None)
    patient_adapter.add_argument("--target-weight-kg", type=float, default=None)
    patient_adapter.add_argument("--target-bmi", type=float, default=None)
    patient_adapter.add_argument("--target-waist-cm", type=float, default=None)
    patient_copy = patient_adapter.add_mutually_exclusive_group()
    patient_copy.add_argument("--copy-inputs", dest="copy_inputs", action="store_true", default=False)
    patient_copy.add_argument("--no-copy-inputs", dest="copy_inputs", action="store_false")
    patient_adapter.add_argument("--report", default="outputs/reports/patient_phantom_input_adapter_stage001.md")

    gi_stage = subparsers.add_parser(
        "stage-gi-segmentation",
        help="Validate and normalize a real GI/bowel/colon/small-intestine segmentation for product-case replacement.",
    )
    gi_stage.add_argument("--ct", required=True, help="Primary CT NIfTI on the target phantom grid.")
    gi_stage.add_argument("--gi-seg", required=True, help="GI multilabel NIfTI or directory of binary organ masks.")
    gi_stage.add_argument("--gi-labelmap", default="configs/labelmaps/gi_tract.yaml")
    gi_stage.add_argument("--output-dir", default="outputs/digital/gi_segmentation_staging")
    gi_stage.add_argument("--case-id", default="gi_segmentation_stage001")
    gi_stage.add_argument("--report", default="outputs/reports/gi_segmentation_staging_stage001.md")
    gi_stage.add_argument(
        "--require-targets",
        nargs="*",
        default=["small_bowel", "colon", "rectum"],
        help="GI targets required for full readiness. Defaults to small_bowel colon rectum.",
    )

    gi_auto = subparsers.add_parser(
        "auto-stage-gi-segmentation",
        help="Run or reuse a supported auto-segmenter output folder, then stage real GI masks for product-case replacement.",
    )
    gi_auto.add_argument("--ct", required=True, help="Primary CT NIfTI on the target phantom grid.")
    gi_auto.add_argument("--segmenter", choices=["totalsegmentator"], default="totalsegmentator")
    gi_auto.add_argument("--segmenter-output-dir", default=None, help="Existing or target output folder for segmenter masks.")
    gi_auto.add_argument("--segmenter-executable", default=None, help="Optional path/name overriding auto-detected segmenter executable.")
    gi_auto.add_argument("--segmenter-args", default="", help="Extra segmenter arguments, for example: '--fast'.")
    gi_auto.add_argument("--gi-labelmap", default="configs/labelmaps/gi_tract.yaml")
    gi_auto.add_argument("--output-dir", default="outputs/digital/gi_autoseg_bridge")
    gi_auto.add_argument("--case-id", default="gi_autoseg_stage001")
    gi_auto.add_argument("--report", default="outputs/reports/gi_autoseg_bridge_stage001.md")
    gi_auto.add_argument(
        "--require-targets",
        nargs="*",
        default=["small_bowel", "colon", "rectum"],
        help="GI targets required for full readiness. Defaults to small_bowel colon rectum.",
    )
    gi_auto.add_argument("--force-rerun", action="store_true", help="Run the segmenter even if a supported output folder already exists.")
    gi_auto.add_argument("--dry-run", action="store_true", help="Write planned command/report without running the segmenter.")
    gi_auto.add_argument("--timeout-s", type=int, default=None, help="Optional segmenter execution timeout in seconds.")

    patient_build = subparsers.add_parser(
        "run-patient-phantom-build",
        help="Execute a patient phantom build from a patient-input adapter manifest.",
    )
    patient_build.add_argument("--patient-manifest", required=True, help="Patient input adapter manifest YAML.")
    patient_build.add_argument("--case-id", default=None)
    patient_build.add_argument("--output-dir", default="outputs/digital/patient_builds")
    patient_build.add_argument("--organ-labelmap", default=None)
    patient_build.add_argument("--materials", default=None)
    patient_build.add_argument("--baseline-graph", default=None)
    patient_build.add_argument("--sample-step-mm", type=float, default=0.75)
    patient_build.add_argument("--vessel-wall-thickness-mm", type=float, default=2.0)
    patient_build.add_argument("--arterial-inlet-flow-ml-s", type=float, default=80.0)
    patient_build.add_argument("--heart-rate-bpm", type=float, default=60.0)
    patient_build.add_argument("--organ-label-mode", choices=["auto", "ct-org", "material", "btcv"], default="auto")
    patient_build.add_argument("--correct-bone-conflicts", action="store_true")
    patient_build.add_argument("--bone-clearance-mm", type=float, default=8.0)
    patient_build.add_argument("--dry-run", action="store_true")
    patient_build.add_argument("--allow-template-vessels", action="store_true")
    rt_run = patient_build.add_mutually_exclusive_group()
    rt_run.add_argument("--run-rt", dest="run_rt", action="store_true", default=True)
    rt_run.add_argument("--skip-rt", dest="run_rt", action="store_false")
    dicom_patient = patient_build.add_mutually_exclusive_group()
    dicom_patient.add_argument("--export-dicom", dest="export_dicom", action="store_true", default=False)
    dicom_patient.add_argument("--skip-dicom", dest="export_dicom", action="store_false")
    patient_build.add_argument("--report", default="outputs/reports/patient_phantom_build_executor_stage001.md")

    patient_build_qa = subparsers.add_parser(
        "qa-patient-phantom-build",
        help="Gate a completed patient phantom build using anatomy, vessel, flow, and RT QA specs.",
    )
    patient_build_qa.add_argument("--build-manifest", required=True, help="Patient build executor manifest YAML.")
    patient_build_qa.add_argument("--case-id", default=None)
    patient_build_qa.add_argument("--output-dir", default="outputs/qa/patient_builds")
    patient_build_qa.add_argument("--max-organ-fail-edges", type=int, default=0)
    patient_build_qa.add_argument("--max-organ-review-edges", type=int, default=0)
    patient_build_qa.add_argument("--max-radius-fail-edges", type=int, default=0)
    patient_build_qa.add_argument("--max-radius-review-edges", type=int, default=0)
    patient_build_qa.add_argument("--expected-lumen-components", type=int, default=1)
    patient_build_qa.add_argument("--max-overlap-after-cleanup", type=int, default=0)
    patient_build_qa.add_argument("--max-outside-body-fraction", type=float, default=0.0)
    patient_build_qa.add_argument("--max-flow-mass-residual-ml-s", type=float, default=1e-4)
    patient_build_qa.add_argument("--report", default="outputs/reports/patient_phantom_build_qa_gate_stage001.md")

    product_case = subparsers.add_parser(
        "build-product-case",
        aliases=["build-case"],
        help="Run a user-facing product case package from CT/labels or an existing patient build manifest.",
    )
    product_case.add_argument("--input-ct", default=None, help="Primary CT NIfTI or DICOM directory.")
    product_case.add_argument("--input-cta", default=None, help="Optional CTA NIfTI or DICOM directory.")
    product_case.add_argument("--input-ctv", default=None, help="Optional CTV NIfTI or DICOM directory.")
    product_case.add_argument("--organ-seg", default=None, help="Organ/material segmentation NIfTI.")
    product_case.add_argument("--gi-seg", default=None, help="Optional co-registered GI/bowel/colon/small-intestine segmentation NIfTI or mask directory.")
    product_case.add_argument("--vessel-seg", default=None, help="Arterial/venous vessel segmentation NIfTI.")
    product_case.add_argument("--existing-build-manifest", default=None, help="Package an existing patient build manifest without rebuilding.")
    product_case.add_argument("--patient-id", default="patient_demo")
    product_case.add_argument("--case-id", default=None)
    product_case.add_argument("--output-dir", default="outputs/product_cases")
    product_case.add_argument("--organ-labelmap", default=str(DEFAULT_CT_ORG_LABELMAP))
    product_case.add_argument("--gi-labelmap", default="configs/labelmaps/gi_tract.yaml")
    product_case.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    product_case.add_argument("--baseline-graph", default=None)
    product_case.add_argument("--baseline-combined-spec", default=None)
    product_case.add_argument("--approved-set-manifest", default=None)
    product_case.add_argument("--target-height-cm", type=float, default=None)
    product_case.add_argument("--target-weight-kg", type=float, default=None)
    product_case.add_argument("--target-bmi", type=float, default=None)
    product_case.add_argument("--target-waist-cm", type=float, default=None)
    product_copy = product_case.add_mutually_exclusive_group()
    product_copy.add_argument("--copy-inputs", dest="copy_inputs", action="store_true", default=False)
    product_copy.add_argument("--no-copy-inputs", dest="copy_inputs", action="store_false")
    product_case.add_argument("--allow-template-vessels", action="store_true")
    product_case.add_argument("--dry-run", action="store_true")
    product_rt = product_case.add_mutually_exclusive_group()
    product_rt.add_argument("--run-rt", dest="run_rt", action="store_true", default=True)
    product_rt.add_argument("--skip-rt", dest="run_rt", action="store_false")
    product_dicom = product_case.add_mutually_exclusive_group()
    product_dicom.add_argument("--export-dicom", dest="export_dicom", action="store_true", default=False)
    product_dicom.add_argument("--skip-dicom", dest="export_dicom", action="store_false")
    product_case.add_argument("--sample-step-mm", type=float, default=0.75)
    product_case.add_argument("--vessel-wall-thickness-mm", type=float, default=2.0)
    product_case.add_argument("--arterial-inlet-flow-ml-s", type=float, default=80.0)
    product_case.add_argument("--heart-rate-bpm", type=float, default=60.0)
    product_case.add_argument("--organ-label-mode", choices=["auto", "ct-org", "material", "btcv"], default="auto")
    product_case.add_argument("--correct-bone-conflicts", action="store_true")
    product_case.add_argument("--bone-clearance-mm", type=float, default=8.0)
    product_qa = product_case.add_mutually_exclusive_group()
    product_qa.add_argument("--run-qa", dest="run_qa", action="store_true", default=True)
    product_qa.add_argument("--skip-qa", dest="run_qa", action="store_false")
    product_case.add_argument("--qa-expected-lumen-components", type=int, default=1)
    product_render = product_case.add_mutually_exclusive_group()
    product_render.add_argument("--render-3d", dest="render_3d", action="store_true", default=True)
    product_render.add_argument("--skip-render-3d", dest="render_3d", action="store_false")
    product_vessel_render = product_case.add_mutually_exclusive_group()
    product_vessel_render.add_argument("--render-vessel-visible", dest="render_vessel_visible", action="store_true", default=True)
    product_vessel_render.add_argument("--skip-vessel-visible", dest="render_vessel_visible", action="store_false")
    product_case.add_argument("--existing-render-preview", default=None, help="Existing CAD-style 3D PNG to link into the product package.")
    product_case.add_argument("--existing-render-scene", default=None, help="Existing 3D render scene YAML/STL spec to link into the product package.")
    product_case.add_argument("--render-target-max-faces", type=int, default=90_000)
    product_case.add_argument("--report", default=None)

    patient_case_adapter = subparsers.add_parser(
        "run-patient-case-adapter",
        aliases=["adapt-patient-case"],
        help="Score a user CT case against the morph library, select the closest profile, then run build/QA/render.",
    )
    patient_case_adapter.add_argument("--input-ct", required=True, help="Primary CT NIfTI.")
    patient_case_adapter.add_argument("--input-cta", default=None, help="Optional CTA NIfTI or DICOM directory.")
    patient_case_adapter.add_argument("--input-ctv", default=None, help="Optional CTV NIfTI or DICOM directory.")
    patient_case_adapter.add_argument("--organ-seg", default=None, help="Optional organ/material segmentation NIfTI on the CT grid.")
    patient_case_adapter.add_argument("--vessel-seg", default=None, help="Optional arterial/venous vessel segmentation NIfTI on the CT grid.")
    patient_case_adapter.add_argument("--patient-id", default="patient_demo")
    patient_case_adapter.add_argument("--case-id", default=None)
    patient_case_adapter.add_argument("--output-dir", default="outputs/patient_case_adapter")
    patient_case_adapter.add_argument(
        "--profile-spec-glob",
        default="outputs/experiments/profile_sweep/**/anthropometry/*_anthro_morph_spec_v001.yaml",
        help="Glob for existing anthropometric morph specs.",
    )
    patient_case_adapter.add_argument(
        "--profile-metrics-csv",
        default=None,
        help="Optional profile metrics CSV, for example from run-deformation-match-experiment.",
    )
    patient_case_grid = patient_case_adapter.add_mutually_exclusive_group()
    patient_case_grid.add_argument("--include-height-grid", dest="include_height_grid", action="store_true", default=True)
    patient_case_grid.add_argument("--no-height-grid", dest="include_height_grid", action="store_false")
    patient_case_adapter.add_argument("--organ-labelmap", default=str(DEFAULT_CT_ORG_LABELMAP))
    patient_case_adapter.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    patient_case_adapter.add_argument("--baseline-graph", default=None)
    patient_case_adapter.add_argument("--baseline-combined-spec", default=None)
    patient_case_adapter.add_argument("--approved-set-manifest", default=None)
    patient_case_adapter.add_argument("--target-height-cm", type=float, default=None)
    patient_case_adapter.add_argument("--target-weight-kg", type=float, default=None)
    patient_case_adapter.add_argument("--target-bmi", type=float, default=None)
    patient_case_adapter.add_argument("--target-waist-cm", type=float, default=None)
    patient_case_copy = patient_case_adapter.add_mutually_exclusive_group()
    patient_case_copy.add_argument("--copy-inputs", dest="copy_inputs", action="store_true", default=False)
    patient_case_copy.add_argument("--no-copy-inputs", dest="copy_inputs", action="store_false")
    patient_case_template = patient_case_adapter.add_mutually_exclusive_group()
    patient_case_template.add_argument("--allow-template-vessels", dest="allow_template_vessels", action="store_true", default=True)
    patient_case_template.add_argument("--no-template-vessels", dest="allow_template_vessels", action="store_false")
    patient_case_adapter.add_argument("--dry-run", action="store_true")
    patient_case_rt = patient_case_adapter.add_mutually_exclusive_group()
    patient_case_rt.add_argument("--run-rt", dest="run_rt", action="store_true", default=True)
    patient_case_rt.add_argument("--skip-rt", dest="run_rt", action="store_false")
    patient_case_dicom = patient_case_adapter.add_mutually_exclusive_group()
    patient_case_dicom.add_argument("--export-dicom", dest="export_dicom", action="store_true", default=False)
    patient_case_dicom.add_argument("--skip-dicom", dest="export_dicom", action="store_false")
    patient_case_adapter.add_argument("--sample-step-mm", type=float, default=0.75)
    patient_case_adapter.add_argument("--vessel-wall-thickness-mm", type=float, default=2.0)
    patient_case_adapter.add_argument("--arterial-inlet-flow-ml-s", type=float, default=80.0)
    patient_case_adapter.add_argument("--heart-rate-bpm", type=float, default=60.0)
    patient_case_adapter.add_argument("--organ-label-mode", choices=["auto", "ct-org", "material", "btcv"], default="auto")
    patient_case_adapter.add_argument("--correct-bone-conflicts", action="store_true")
    patient_case_adapter.add_argument("--bone-clearance-mm", type=float, default=8.0)
    patient_case_qa = patient_case_adapter.add_mutually_exclusive_group()
    patient_case_qa.add_argument("--run-qa", dest="run_qa", action="store_true", default=True)
    patient_case_qa.add_argument("--skip-qa", dest="run_qa", action="store_false")
    patient_case_adapter.add_argument("--qa-expected-lumen-components", type=int, default=1)
    patient_case_render = patient_case_adapter.add_mutually_exclusive_group()
    patient_case_render.add_argument("--render-3d", dest="render_3d", action="store_true", default=True)
    patient_case_render.add_argument("--skip-render-3d", dest="render_3d", action="store_false")
    patient_case_adapter.add_argument("--render-target-max-faces", type=int, default=90_000)
    patient_case_adapter.add_argument("--score-only", action="store_true")
    patient_case_adapter.add_argument("--report", default=None)

    product_release = subparsers.add_parser(
        "build-product-release-package",
        help="Create a disk-light release folder from a product case manifest.",
    )
    product_release.add_argument("--product-manifest", required=True, help="Product case manifest YAML.")
    product_release.add_argument("--release-id", default="product_mvp_rc1")
    product_release.add_argument("--output-dir", default="outputs/releases/product_cases")
    product_copy = product_release.add_mutually_exclusive_group()
    product_copy.add_argument("--copy-small-artifacts", dest="copy_small_artifacts", action="store_true", default=True)
    product_copy.add_argument("--index-only", dest="copy_small_artifacts", action="store_false")
    product_release.add_argument("--large-threshold-bytes", type=int, default=25_000_000)
    product_release.add_argument(
        "--command",
        dest="command_lines",
        action="append",
        default=[],
        help="Upstream command line to include in COMMANDS.md. Repeatable.",
    )
    product_release.add_argument(
        "--supplemental-artifact",
        dest="supplemental_artifacts",
        action="append",
        default=[],
        help="Extra file or directory to include in the release artifact index. Repeatable.",
    )
    product_release.add_argument("--report", default=None)

    release_case = subparsers.add_parser(
        "build-product-release-case",
        help="Run product case generation, QA, release packaging, and a workflow report in one command.",
    )
    release_case.add_argument("--input-ct", default=None, help="Primary CT NIfTI or DICOM directory.")
    release_case.add_argument("--input-cta", default=None, help="Optional CTA NIfTI or DICOM directory.")
    release_case.add_argument("--input-ctv", default=None, help="Optional CTV NIfTI or DICOM directory.")
    release_case.add_argument("--organ-seg", default=None, help="Organ/material segmentation NIfTI.")
    release_case.add_argument("--vessel-seg", default=None, help="Arterial/venous vessel segmentation NIfTI.")
    release_case.add_argument("--existing-build-manifest", default=None, help="Package an existing patient build manifest without rebuilding.")
    release_case.add_argument("--patient-id", default="patient_demo")
    release_case.add_argument("--case-id", default=None)
    release_case.add_argument("--output-dir", default="outputs/product_release_cases")
    release_case.add_argument("--product-output-dir", default="outputs/product_cases")
    release_case.add_argument("--release-output-dir", default="outputs/releases/product_cases")
    release_case.add_argument("--release-id", default="product_case_rc1")
    release_case.add_argument("--organ-labelmap", default=str(DEFAULT_CT_ORG_LABELMAP))
    release_case.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    release_case.add_argument("--baseline-graph", default=None)
    release_case.add_argument("--baseline-combined-spec", default=None)
    release_case.add_argument("--approved-set-manifest", default=None)
    release_case.add_argument("--target-height-cm", type=float, default=None)
    release_case.add_argument("--target-weight-kg", type=float, default=None)
    release_case.add_argument("--target-bmi", type=float, default=None)
    release_case.add_argument("--target-waist-cm", type=float, default=None)
    release_case_copy = release_case.add_mutually_exclusive_group()
    release_case_copy.add_argument("--copy-inputs", dest="copy_inputs", action="store_true", default=False)
    release_case_copy.add_argument("--no-copy-inputs", dest="copy_inputs", action="store_false")
    release_case.add_argument("--allow-template-vessels", action="store_true")
    release_case.add_argument("--dry-run", action="store_true")
    release_case_rt = release_case.add_mutually_exclusive_group()
    release_case_rt.add_argument("--run-rt", dest="run_rt", action="store_true", default=True)
    release_case_rt.add_argument("--skip-rt", dest="run_rt", action="store_false")
    release_case_dicom = release_case.add_mutually_exclusive_group()
    release_case_dicom.add_argument("--export-dicom", dest="export_dicom", action="store_true", default=False)
    release_case_dicom.add_argument("--skip-dicom", dest="export_dicom", action="store_false")
    release_case.add_argument("--sample-step-mm", type=float, default=0.75)
    release_case.add_argument("--vessel-wall-thickness-mm", type=float, default=2.0)
    release_case.add_argument("--arterial-inlet-flow-ml-s", type=float, default=80.0)
    release_case.add_argument("--heart-rate-bpm", type=float, default=60.0)
    release_case.add_argument("--organ-label-mode", choices=["auto", "ct-org", "material", "btcv"], default="auto")
    release_case.add_argument("--correct-bone-conflicts", action="store_true")
    release_case.add_argument("--bone-clearance-mm", type=float, default=8.0)
    release_case_qa = release_case.add_mutually_exclusive_group()
    release_case_qa.add_argument("--run-qa", dest="run_qa", action="store_true", default=True)
    release_case_qa.add_argument("--skip-qa", dest="run_qa", action="store_false")
    release_case.add_argument("--qa-expected-lumen-components", type=int, default=1)
    release_case_render = release_case.add_mutually_exclusive_group()
    release_case_render.add_argument("--render-3d", dest="render_3d", action="store_true", default=True)
    release_case_render.add_argument("--skip-render-3d", dest="render_3d", action="store_false")
    release_case.add_argument("--existing-render-preview", default=None, help="Existing CAD-style 3D PNG to link into the product package.")
    release_case.add_argument("--existing-render-scene", default=None, help="Existing 3D render scene YAML/STL spec to link into the product package.")
    release_case.add_argument("--render-target-max-faces", type=int, default=90_000)
    release_case_release_copy = release_case.add_mutually_exclusive_group()
    release_case_release_copy.add_argument(
        "--copy-small-release-artifacts",
        dest="copy_small_release_artifacts",
        action="store_true",
        default=True,
    )
    release_case_release_copy.add_argument("--index-release-only", dest="copy_small_release_artifacts", action="store_false")
    release_case.add_argument("--release-large-threshold-bytes", type=int, default=25_000_000)
    release_case.add_argument(
        "--release-command",
        dest="release_command_lines",
        action="append",
        default=[],
        help="Upstream command line to include in the release COMMANDS.md. Repeatable.",
    )
    release_case.add_argument(
        "--supplemental-artifact",
        dest="supplemental_artifacts",
        action="append",
        default=[],
        help="Extra file or directory to include in the release artifact index. Repeatable.",
    )
    release_case.add_argument("--report", default=None)

    profile_compare = subparsers.add_parser(
        "build-profile-rerun-comparison-atlas",
        help="Build a side-by-side status and delta atlas for a profile-specific morph/flow/RT rerun.",
    )
    profile_compare.add_argument("--case-id", default="ct_org_label_population8_bmi32_waist110_height175")
    profile_compare.add_argument("--profile-id", default="bmi32_waist110_height175")
    profile_compare.add_argument("--output-dir", default="outputs/reports/profile_rerun_comparison")
    profile_compare.add_argument("--report", default="outputs/reports/profile_rerun_comparison_stage001.md")
    profile_compare.add_argument("--profile-adapter-spec", default="outputs/digital/user_profile_adapter/ct_org_label_population8_bmi32_waist110_height175_bmi32_waist110_height175_profile_adapter_v001.yaml")
    profile_compare.add_argument("--anthropometric-spec", default="outputs/digital/anthropometric_morph/bmi32_waist110_height175/ct_org_label_population8_bmi32_waist110_height175_anthro_morph_spec_v001.yaml")
    profile_compare.add_argument("--reference-vascular-spec", default="outputs/digital/variant_flow/mode01_pos/vascular_network_voxelized/ct_org_label_population8_pca_modes_stage001_mode01_pos_vascular_network_voxelized_spec_v001.yaml")
    profile_compare.add_argument("--profile-vascular-spec", default="outputs/digital/profile_flow/bmi32_waist110_height175/vascular_network_voxelized/ct_org_label_population8_bmi32_waist110_height175_vascular_network_voxelized_spec_v001.yaml")
    profile_compare.add_argument("--reference-flow-spec", default="outputs/sim/variant_flow/mode01_pos/flow_coupled_pulsatile/ct_org_label_population8_pca_modes_stage001_mode01_pos_coupled_pulsatile_flow_model_v001.yaml")
    profile_compare.add_argument("--profile-flow-spec", default="outputs/sim/profile_flow/bmi32_waist110_height175/flow_coupled_pulsatile/ct_org_label_population8_bmi32_waist110_height175_coupled_pulsatile_flow_model_v001.yaml")
    profile_compare.add_argument("--reference-spatial-dose-spec", default="outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/ct_org_label_population8_pca_modes_stage001_mode01_pos_rt_spatial_flow_dose_model_spec_v001.yaml")
    profile_compare.add_argument("--profile-spatial-dose-spec", default="outputs/radiotherapy/spatial_rt_flow_dose/bmi32_waist110_height175/ct_org_label_population8_bmi32_waist110_height175_rt_spatial_flow_dose_model_spec_v001.yaml")
    profile_compare.add_argument("--reference-gamma-spec", default="outputs/radiotherapy/spatial_rt_flow_dose/mode01_pos/dose_gamma_qa/ct_org_label_population8_pca_modes_stage001_mode01_pos_spatial_flow_dose_gamma_qa_spec_v001.yaml")
    profile_compare.add_argument("--profile-gamma-spec", default="outputs/radiotherapy/spatial_rt_flow_dose/bmi32_waist110_height175/dose_gamma_qa/ct_org_label_population8_bmi32_waist110_height175_spatial_flow_dose_dose_gamma_qa_spec_v001.yaml")

    profile_sweep = subparsers.add_parser(
        "build-profile-sweep",
        help="Run a multi-profile BMI/waist anthropometric sweep through morph, flow, and RT-flow QA.",
    )
    profile_sweep.add_argument("--sweep-id", default="ct_org_profile_sweep_stage001")
    profile_sweep.add_argument("--output-dir", default="outputs/experiments/profile_sweep")
    profile_sweep.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Profile spec as profile_id:bmi:waist_cm:height_cm. Repeat to override defaults.",
    )
    profile_sweep.add_argument("--combined-spec", default="outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml")
    profile_sweep.add_argument("--baseline-graph", default="outputs/digital/vascular_network/ct_org_case0_imagetbad_case125_vascular_network_graph_v001.yaml")
    profile_sweep.add_argument("--baseline-labels", default="outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_material_labels_blood_v001.nii.gz")
    profile_sweep.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    profile_sweep.add_argument("--baseline-height-cm", type=float, default=170.0)
    profile_sweep.add_argument("--baseline-bmi", type=float, default=24.0)
    profile_sweep.add_argument("--arterial-inlet-flow-ml-s", type=float, default=80.0)
    profile_sweep.add_argument("--high-bmi-waist-threshold-cm", type=float, default=115.0)
    profile_sweep.add_argument("--high-bmi-xy-padding-voxels", type=int, default=96)
    profile_sweep.add_argument("--padding-transition-margin-cm", type=float, default=5.0)
    dicom_sweep = profile_sweep.add_mutually_exclusive_group()
    dicom_sweep.add_argument("--export-dicom", dest="export_dicom", action="store_true", default=False)
    dicom_sweep.add_argument("--skip-dicom", dest="export_dicom", action="store_false")
    profile_sweep.add_argument("--gamma-random-subset", type=int, default=25000)
    profile_sweep.add_argument("--report", default="outputs/reports/profile_sweep_stage001.md")

    profile_prescription = subparsers.add_parser(
        "build-profile-prescription",
        help="Convert profile sweep metrics into a BMI/waist/height operating-envelope prescription.",
    )
    profile_prescription.add_argument("--metrics-csv", default="outputs/experiments/profile_sweep/ct_org_profile_sweep_stage001_profile_sweep_metrics_v001.csv")
    profile_prescription.add_argument("--profile-id", default="bmi35_waist118_height175")
    profile_prescription.add_argument("--case-id", default=None)
    profile_prescription.add_argument("--output-dir", default="outputs/digital/profile_prescriptions")
    profile_prescription.add_argument("--target-height-cm", type=float, default=175.0)
    profile_prescription.add_argument("--target-weight-kg", type=float, default=None)
    profile_prescription.add_argument("--target-bmi", type=float, default=35.0)
    profile_prescription.add_argument("--target-waist-cm", type=float, default=118.0)
    profile_prescription.add_argument("--baseline-bmi", type=float, default=24.0)
    profile_prescription.add_argument("--baseline-waist-cm", type=float, default=None)
    profile_prescription.add_argument("--waist-tolerance-cm", type=float, default=3.0)
    profile_prescription.add_argument("--bmi-tolerance", type=float, default=2.0)
    profile_prescription.add_argument("--height-tolerance-cm", type=float, default=10.0)
    profile_prescription.add_argument("--high-bmi-waist-threshold-cm", type=float, default=115.0)
    profile_prescription.add_argument("--high-bmi-xy-padding-voxels", type=int, default=96)
    profile_prescription.add_argument("--padding-transition-margin-cm", type=float, default=5.0)
    profile_prescription.add_argument("--combined-spec", default="outputs/digital/combined/ct_org_case0_imagetbad_case125_combined_spec_v001.yaml")
    profile_prescription.add_argument("--report", default="outputs/reports/profile_operating_prescription_stage001.md")

    profile_envelope = subparsers.add_parser(
        "build-profile-envelope",
        help="Merge profile sweep metrics into a consolidated operating envelope and optional prescription validation report.",
    )
    profile_envelope.add_argument(
        "--metrics-csv",
        action="append",
        required=True,
        help="Profile sweep metrics CSV. Repeat to merge additional validated profiles; later duplicates override earlier rows.",
    )
    profile_envelope.add_argument("--envelope-id", default="ct_org_profile_envelope_stage001")
    profile_envelope.add_argument("--output-dir", default="outputs/experiments/profile_envelope")
    profile_envelope.add_argument("--prescription-yaml", default=None)
    profile_envelope.add_argument("--report", default="outputs/reports/profile_operating_envelope_stage001.md")

    profile_planner = subparsers.add_parser(
        "plan-next-profiles",
        help="Rank the next BMI/waist profiles to validate from a consolidated profile envelope.",
    )
    profile_planner.add_argument("--metrics-csv", default="outputs/experiments/profile_envelope/ct_org_profile_envelope_with_bmi35_stage001_consolidated_profile_metrics_v001.csv")
    profile_planner.add_argument("--plan-id", default="ct_org_next_profile_plan_stage001")
    profile_planner.add_argument("--output-dir", default="outputs/experiments/profile_planning")
    profile_planner.add_argument("--high-bmi-waist-threshold-cm", type=float, default=115.0)
    profile_planner.add_argument("--high-bmi-xy-padding-voxels", type=int, default=96)
    profile_planner.add_argument("--padding-transition-margin-cm", type=float, default=5.0)
    profile_planner.add_argument("--transition-margin-cm", type=float, default=1.0)
    profile_planner.add_argument("--min-distance-from-existing-cm", type=float, default=1.5)
    profile_planner.add_argument("--max-candidates", type=int, default=5)
    profile_planner.add_argument("--gamma-random-subset", type=int, default=25000)
    profile_planner.add_argument("--report", default="outputs/reports/next_profile_validation_plan_stage001.md")

    experiment = subparsers.add_parser(
        "run-phantom-experiment-set",
        help="Run a disk-light RT/flow/anatomy comparison across an approved PCA phantom set.",
    )
    experiment.add_argument("--approved-set-manifest", required=True, help="Approved PCA phantom-set manifest YAML.")
    experiment.add_argument("--rt-planning-spec", default=None, help="Optional RT planning bundle spec YAML.")
    experiment.add_argument("--dose-gamma-spec", default=None, help="Optional dose gamma QA spec YAML.")
    experiment.add_argument("--flow-model-spec", default=None, help="Optional coupled flow model spec YAML.")
    experiment.add_argument("--case-id", default=None)
    experiment.add_argument("--output-dir", default="outputs/experiments/approved_pca_phantom_set")
    experiment.add_argument("--report", default="outputs/reports/approved_pca_phantom_experiment_set_stage001.md")

    variant_harness = subparsers.add_parser(
        "build-variant-rerun-harness",
        help="Create a disk-light variant-specific RT/flow rerun harness for an approved PCA phantom variant.",
    )
    variant_harness.add_argument("--approved-set-manifest", required=True, help="Approved PCA phantom-set manifest YAML.")
    variant_harness.add_argument("--variant-id", default="mode01_neg")
    variant_harness.add_argument("--baseline-combined-spec", default=None)
    variant_harness.add_argument("--flow-model-spec", default=None)
    variant_harness.add_argument("--case-id", default=None)
    variant_harness.add_argument("--output-dir", default="outputs/experiments/variant_rerun_harness")
    variant_harness.add_argument("--report", default=None)
    material_maps = variant_harness.add_mutually_exclusive_group()
    material_maps.add_argument("--stage-material-maps", dest="stage_material_maps", action="store_true", default=True)
    material_maps.add_argument("--skip-material-maps", dest="stage_material_maps", action="store_false")

    variant_rt_compare = subparsers.add_parser(
        "compare-variant-rt-impact",
        help="Compare a variant-specific RT rerun against the baseline planning and gamma QA bundle.",
    )
    variant_rt_compare.add_argument("--baseline-rt-planning-spec", required=True)
    variant_rt_compare.add_argument("--variant-rt-planning-spec", required=True)
    variant_rt_compare.add_argument("--baseline-gamma-spec", default=None)
    variant_rt_compare.add_argument("--variant-gamma-spec", default=None)
    variant_rt_compare.add_argument("--variant-id", default="mode01_neg")
    variant_rt_compare.add_argument("--case-id", default=None)
    variant_rt_compare.add_argument("--output-dir", default="outputs/experiments/variant_rt_comparison")
    variant_rt_compare.add_argument("--report", default="outputs/reports/variant_rt_impact_comparison_stage001.md")

    render3d = subparsers.add_parser(
        "render-combined-3d",
        help="Export 3D renderable meshes and a transparent preview from a combined material-label map.",
    )
    render3d.add_argument("--combined-labels", required=True, help="Combined material-label NIfTI.")
    render3d.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    render3d.add_argument("--output-dir", default="outputs/render3d/combined")
    render3d.add_argument("--formats", nargs="+", default=["stl", "ply", "obj"])
    render3d.add_argument("--target-max-faces", type=int, default=140_000)
    render3d.add_argument("--report", default="outputs/reports/combined_3d_render_stage001.md")
    render3d.add_argument("--qa-report", default="outputs/reports/combined_3d_render_mesh_qa_stage001.md")

    vascular_render3d = subparsers.add_parser(
        "render-vascular-network-3d",
        help="Export 3D meshes and preview for cleaned voxelized arterial/venous vascular domains.",
    )
    vascular_render3d.add_argument("--context-labels", required=True, help="Combined material-label NIfTI for body/organ context.")
    vascular_render3d.add_argument("--arterial-mask", required=True, help="Cleaned arterial lumen mask NIfTI.")
    vascular_render3d.add_argument("--venous-mask", required=True, help="Cleaned venous lumen mask NIfTI.")
    vascular_render3d.add_argument("--flow-domain-labels", required=True, help="Cleaned flow-domain labels NIfTI.")
    vascular_render3d.add_argument("--vessel-wall-mask", required=True, help="Cleaned vascular-network vessel-wall mask NIfTI.")
    vascular_render3d.add_argument("--vascular-graph", default=None, help="Optional vascular graph YAML for display-enlarged centerlines.")
    vascular_render3d.add_argument("--case-id", default="ct_org_case0_imagetbad_case125")
    vascular_render3d.add_argument("--output-dir", default="outputs/render3d/vascular_network_cleaned")
    vascular_render3d.add_argument("--formats", nargs="+", default=["stl", "ply", "obj"])
    vascular_render3d.add_argument("--target-max-faces", type=int, default=140_000)
    vascular_visible = vascular_render3d.add_mutually_exclusive_group()
    vascular_visible.add_argument("--render-vessel-visible", dest="render_vessel_visible", action="store_true", default=True)
    vascular_visible.add_argument("--skip-vessel-visible", dest="render_vessel_visible", action="store_false")
    vascular_render3d.add_argument("--vessel-display-scale", type=float, default=4.0)
    vascular_render3d.add_argument("--report", default="outputs/reports/vascular_network_3d_render_stage001.md")
    vascular_render3d.add_argument("--qa-report", default="outputs/reports/vascular_network_3d_render_mesh_qa_stage001.md")

    atlas = subparsers.add_parser(
        "render-3d-atlas",
        help="Generate standard multi-view PNGs and a contact-sheet atlas from a 3D scene spec.",
    )
    atlas.add_argument("--scene-spec", required=True, help="3D render scene YAML from render-combined-3d.")
    atlas.add_argument("--case-id", default=None)
    atlas.add_argument("--output-dir", default="outputs/render3d/combined_atlas")
    atlas.add_argument("--report", default="outputs/reports/combined_3d_render_atlas_stage001.md")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "materials-check":
        library = load_material_library(args.materials)
        print(summarize_materials(library))
        return 0

    if args.command == "datasets-list":
        manifest = load_dataset_manifest(args.datasets)
        print(summarize_datasets(manifest))
        return 0

    if args.command == "phase1-summary":
        phase1_config = load_phase1_config(args.phase1)
        library = load_material_library(args.materials)
        manifest = load_dataset_manifest(args.datasets)
        print(build_phase1_summary(phase1_config, library, manifest))
        return 0

    if args.command == "inspect-nifti":
        print(format_nifti_summary(inspect_nifti(args.path)))
        return 0

    if args.command == "export-label-mesh":
        result = export_label_mesh(
            labels_path=args.labels,
            label_id=args.label_id,
            output_path=args.output,
            level=args.level,
        )
        print("Mesh exported")
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0

    if args.command == "mesh-qa":
        results = analyze_meshes(args.meshes)
        report = format_mesh_qa_markdown(results)
        print(report)
        if args.output:
            output = write_mesh_qa_report(results, args.output)
            print(f"\nReport written: {output}")
        return 0

    if args.command == "clean-meshes":
        config = MeshCleanConfig(
            output_dir=Path(args.output_dir),
            formats=tuple(args.formats),
            suffix=args.suffix,
            min_component_faces=args.min_component_faces,
            target_max_faces=args.target_max_faces,
            fill_holes=args.fill_holes,
        )
        results = clean_meshes(args.meshes, config)
        report = format_cleaning_report(results)
        print(report)
        if args.report:
            output = write_cleaning_report(results, args.report)
            print(f"\nCleaning report written: {output}")
        if args.qa_report:
            cleaned_paths = [
                path
                for result in results
                for path in result.output_paths
                if Path(path).suffix.lower() == ".stl"
            ]
            qa_results = analyze_meshes(cleaned_paths)
            qa_output = write_mesh_qa_report(
                qa_results,
                args.qa_report,
                title="Cleaned Mesh QA Report",
            )
            print(f"Cleaned QA report written: {qa_output}")
        return 0

    if args.command == "prepare-vascular-module":
        result = prepare_vascular_module(
            labels_path=args.labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            label_id=args.label_id,
            smooth_sigma=args.smooth_sigma,
            target_max_faces=args.target_max_faces,
            centerline_method=args.centerline_method,
            formats=tuple(args.formats),
            report_path=args.report,
        )
        print(format_vascular_result(result))
        print(f"\nVascular report written: {result.report_path}")
        if args.qa_report:
            qa_meshes = [path for path in result.smoothed_mesh_paths if Path(path).suffix.lower() == ".stl"]
            qa_results = analyze_meshes(qa_meshes)
            qa_output = write_mesh_qa_report(
                qa_results,
                args.qa_report,
                title="Vascular Module Mesh QA Report",
            )
            print(f"Vascular mesh QA report written: {qa_output}")
        return 0

    if args.command == "design-flow-loop":
        result = design_vascular_flow_loop(
            ports_yaml_path=args.ports,
            lumen_mesh_path=args.lumen_mesh,
            output_dir=args.output_dir,
            formats=tuple(args.formats),
            wall_thickness_mm=args.wall_thickness_mm,
            sleeve_length_mm=args.sleeve_length_mm,
            barb_count=args.barb_count,
            barb_height_mm=args.barb_height_mm,
            barb_width_mm=args.barb_width_mm,
            barb_spacing_mm=args.barb_spacing_mm,
            flange_thickness_mm=args.flange_thickness_mm,
            flange_extra_radius_mm=args.flange_extra_radius_mm,
            pressure_tap_diameter_mm=args.pressure_tap_diameter_mm,
            report_path=args.report,
        )
        print(format_flow_loop_result(result))
        print(f"\nFlow-loop design report written: {result.report_path}")
        if args.qa_report:
            qa_meshes = [path for path in result.assembly_paths if Path(path).suffix.lower() == ".stl"]
            qa_results = analyze_meshes(qa_meshes)
            qa_output = write_mesh_qa_report(
                qa_results,
                args.qa_report,
                title="Vascular Flow-Loop Assembly Mesh QA Report",
            )
            print(f"Flow-loop mesh QA report written: {qa_output}")
        return 0

    if args.command == "build-vascular-network-scaffold":
        result = build_vascular_network_scaffold(
            combined_spec_path=args.combined_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            include_venous_return=args.include_venous_return,
            formats=tuple(args.formats),
            body_mesh_path=args.body_mesh,
            report_path=args.report,
        )
        print(format_vascular_network_scaffold_result(result))
        print(f"\nVascular network scaffold report written: {result.report_path}")
        if args.qa_report:
            qa_meshes = [path for path in result.mesh_paths if Path(path).suffix.lower() == ".stl"]
            qa_results = analyze_meshes(qa_meshes)
            qa_output = write_mesh_qa_report(
                qa_results,
                args.qa_report,
                title="Vascular Network Scaffold Mesh QA Report",
            )
            print(f"Vascular network scaffold QA report written: {qa_output}")
        return 0

    if args.command == "build-cta-derived-vascular-graph":
        result = build_cta_derived_vascular_graph(
            baseline_graph_path=args.baseline_graph,
            vascular_mask_path=args.vascular_mask,
            output_dir=args.output_dir,
            case_id=args.case_id,
            use_full_trunk_for_coarse_aorta=args.coarse_aorta_full_trunk,
            report_path=args.report,
        )
        print(format_cta_derived_vascular_graph_result(result))
        print(f"\nCTA-derived vascular graph report written: {result.report_path}")
        return 0

    if args.command == "build-aorta-registration-benchmark":
        result = build_aorta_registration_benchmark(
            manifest_csv_path=args.manifest_csv,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            sample_count=args.sample_count,
            label_value=None if args.use_any_nonzero_label else args.label_value,
            report_path=args.report,
        )
        print(format_aorta_registration_benchmark_result(result))
        print(f"\nAorta registration benchmark report written: {result.report_path}")
        return 0

    if args.command == "apply-learned-aorta-graph":
        result = apply_learned_aorta_to_vascular_graph(
            graph_yaml_path=args.graph,
            aorta_model_path=args.aorta_model,
            output_dir=args.output_dir,
            case_id=args.case_id,
            source_node_id=args.source_node_id,
            target_node_id=args.target_node_id,
            edge_flow_role=args.edge_flow_role,
            point_count=args.point_count,
            radius_scale=args.radius_scale,
            max_radius_mm=args.max_radius_mm,
            minimum_target_span_mm=args.minimum_target_span_mm,
            report_path=args.report,
        )
        print(format_learned_aorta_graph_result(result))
        print(f"\nPopulation-learned aorta graph report written: {result.report_path}")
        return 0

    if args.command == "stage-medseg-abdominal-vasculature":
        result = stage_medseg_abdominal_vasculature(
            raw_dir=args.raw_dir,
            label_config_path=args.label_config,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_medseg_vascular_staging_result(result))
        return 0

    if args.command == "build-labeled-vessel-vascular-graph":
        result = build_labeled_vessel_vascular_graph(
            baseline_graph_path=args.baseline_graph,
            labeled_mask_path=args.labeled_mask,
            label_config_path=args.label_config,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_labeled_vessel_graph_result(result))
        return 0

    if args.command == "build-registered-labeled-vessel-vascular-graph":
        result = build_registered_labeled_vessel_vascular_graph(
            target_graph_path=args.target_graph,
            labeled_mask_path=args.labeled_mask,
            label_config_path=args.label_config,
            target_labels_path=args.target_labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_registered_labeled_vessel_graph_result(result))
        return 0

    if args.command == "build-btcv-branch-anchor-vascular-graph":
        result = build_btcv_branch_anchor_vascular_graph(
            coarse_graph_path=args.coarse_graph,
            anatomy_labels_path=args.anatomy_labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_btcv_branch_anchor_graph_result(result))
        return 0

    if args.command == "tighten-btcv-ivc-trunk":
        result = tighten_btcv_ivc_trunk(
            graph_path=args.graph,
            anatomy_labels_path=args.anatomy_labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            point_count=args.point_count,
            report_path=args.report,
        )
        print(format_btcv_ivc_tightening_result(result))
        return 0

    if args.command == "deform-vascular-graph-for-variant":
        result = deform_vascular_graph_for_variant(
            baseline_graph_path=args.baseline_graph,
            baseline_labels_path=args.baseline_labels,
            variant_labels_path=args.variant_labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            variant_id=args.variant_id,
            report_path=args.report,
        )
        print(format_variant_graph_deformation_result(result))
        print(f"\nVariant vascular graph deformation report written: {result.report_path}")
        return 0

    if args.command == "voxelize-vascular-network":
        result = voxelize_vascular_network(
            graph_yaml_path=args.graph,
            combined_labels_path=args.combined_labels,
            materials_path=args.materials,
            output_dir=args.output_dir,
            case_id=args.case_id,
            source_ct_path=args.source_ct,
            body_mask_path=args.body_mask,
            sample_step_mm=args.sample_step_mm,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            contrast_mode=args.contrast_mode,
            collision_cleanup=args.collision_cleanup,
            clip_to_body=args.clip_to_body,
            write_material_volumes=args.write_material_volumes,
            report_path=args.report,
        )
        print(format_vascular_network_voxelization_result(result))
        print(f"\nVascular network voxelization report written: {result.report_path}")
        return 0

    if args.command == "build-btcv-coarse-vessel-graph":
        result = build_btcv_coarse_vessel_graph(
            graph_yaml_path=args.graph_yaml,
            output_dir=args.output_dir,
            case_id=args.case_id,
            arterial_radius_mm=args.arterial_radius_mm,
            venous_radius_mm=args.venous_radius_mm,
            report_path=args.report,
        )
        print(format_coarse_vessel_graph_result(result))
        print(f"\nBTCV coarse vessel graph report written: {result.report_path}")
        return 0

    if args.command == "validate-vessel-organ-anatomy":
        result = validate_vessel_organ_anatomy(
            voxelized_spec_path=args.voxelized_spec,
            graph_yaml_path=args.graph,
            anatomy_labels_path=args.anatomy_labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            sample_step_mm=args.sample_step_mm,
            report_path=args.report,
        )
        print(format_vessel_anatomy_validation_result(result))
        print(f"\nVessel-organ anatomy validation report written: {result.report_path}")
        return 0

    if args.command == "correct-vessel-bone-conflicts":
        result = correct_vessel_bone_conflicts(
            graph_yaml_path=args.graph,
            anatomy_labels_path=args.anatomy_labels,
            edge_metrics_csv_path=args.edge_metrics,
            output_dir=args.output_dir,
            case_id=args.case_id,
            clearance_mm=args.clearance_mm,
            edge_bone_review_threshold=args.edge_bone_review_threshold,
            max_node_shift_mm=args.max_node_shift_mm,
            max_point_shift_mm=args.max_point_shift_mm,
            smooth_iterations=args.smooth_iterations,
            report_path=args.report,
        )
        print(format_vessel_anatomy_correction_result(result))
        print(f"\nVessel anatomy correction report written: {result.report_path}")
        return 0

    if args.command == "reroute-vessel-edge-around-bone":
        result = reroute_vessel_edge_around_bone(
            graph_yaml_path=args.graph,
            anatomy_labels_path=args.anatomy_labels,
            edge_id=args.edge_id,
            output_dir=args.output_dir,
            case_id=args.case_id,
            clearance_mm=args.clearance_mm,
            max_detour_mm=args.max_detour_mm,
            detour_step_mm=args.detour_step_mm,
            sample_step_mm=args.sample_step_mm,
            resample_step_mm=args.resample_step_mm,
            max_point_shift_mm=args.max_point_shift_mm,
            smooth_iterations=args.smooth_iterations,
            report_path=args.report,
        )
        print(format_vessel_edge_reroute_result(result))
        print(f"\nVessel edge reroute report written: {result.report_path}")
        return 0

    if args.command == "repair-vessel-outside-body-margin":
        result = repair_vessel_outside_body_margin(
            graph_yaml_path=args.graph,
            anatomy_labels_path=args.anatomy_labels,
            edge_ids=tuple(args.edge_ids),
            output_dir=args.output_dir,
            case_id=args.case_id,
            sample_step_mm=args.sample_step_mm,
            body_margin_mm=args.body_margin_mm,
            min_radius_mm=args.min_radius_mm,
            max_profile_points=args.max_profile_points,
            report_path=args.report,
        )
        print(format_vessel_outside_body_repair_result(result))
        print(f"\nVessel outside-body repair report written: {result.report_path}")
        return 0

    if args.command == "validate-vessel-radius-anatomy":
        result = validate_vessel_radius_anatomy(
            voxelized_spec_path=args.voxelized_spec,
            graph_yaml_path=args.graph,
            anatomy_labels_path=args.anatomy_labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            sample_step_mm=args.sample_step_mm,
            scaled_radius_factor=args.scaled_radius_factor,
            review_lumen_bone_fraction=args.review_lumen_bone_fraction,
            fail_lumen_bone_fraction=args.fail_lumen_bone_fraction,
            report_path=args.report,
        )
        print(format_vessel_radius_validation_result(result))
        print(f"\nVessel radius-aware anatomy validation report written: {result.report_path}")
        return 0

    if args.command == "tune-vessel-radii-against-bone":
        result = tune_vessel_radii_against_bone(
            graph_yaml_path=args.graph,
            anatomy_labels_path=args.anatomy_labels,
            radius_metrics_csv_path=args.radius_metrics,
            output_dir=args.output_dir,
            case_id=args.case_id,
            edge_ids=tuple(args.edge_ids),
            tune_review_edges_only=args.tune_review_edges_only,
            bone_clearance_mm=args.bone_clearance_mm,
            sample_step_mm=args.sample_step_mm,
            max_profile_points=args.max_profile_points,
            smooth_iterations=args.smooth_iterations,
            min_radius_mm=args.min_radius_mm,
            branch_max_radius_mm=args.branch_max_radius_mm,
            arterial_trunk_min_radius_mm=args.arterial_trunk_min_radius_mm,
            arterial_trunk_max_radius_mm=args.arterial_trunk_max_radius_mm,
            venous_trunk_min_radius_mm=args.venous_trunk_min_radius_mm,
            venous_trunk_max_radius_mm=args.venous_trunk_max_radius_mm,
            report_path=args.report,
        )
        print(format_vessel_radius_tuning_result(result))
        print(f"\nVessel radius tuning report written: {result.report_path}")
        return 0

    if args.command == "repair-vascular-domain-connectivity":
        result = repair_vascular_domain_connectivity(
            voxelized_spec_path=args.voxelized_spec,
            graph_yaml_path=args.graph,
            combined_labels_path=args.combined_labels,
            materials_path=args.materials,
            output_dir=args.output_dir,
            case_id=args.case_id,
            sample_step_mm=args.sample_step_mm,
            seed_search_radius_voxels=args.seed_search_radius_voxels,
            max_unseeded_component_voxels=args.max_unseeded_component_voxels,
            connector_radius_mm=args.connector_radius_mm,
            connect_seeded_components=args.connect_seeded_components,
            contrast_mode=args.contrast_mode,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            write_material_volumes=args.write_material_volumes,
            report_path=args.report,
        )
        print(format_vascular_domain_connectivity_repair_result(result))
        print(f"\nVascular domain connectivity repair report written: {result.report_path}")
        return 0

    if args.command == "build-flow-boundary-package":
        result = build_flow_boundary_package(
            voxelized_spec_path=args.voxelized_spec,
            graph_yaml_path=args.graph,
            output_dir=args.output_dir,
            case_id=args.case_id,
            arterial_inlet_flow_ml_s=args.arterial_inlet_flow_ml_s,
            nominal_outlet_pressure_drop_pa=args.nominal_outlet_pressure_drop_pa,
            venous_outlet_pressure_pa=args.venous_outlet_pressure_pa,
            boundary_slab_thickness_mm=args.boundary_slab_thickness_mm,
            report_path=args.report,
        )
        print(format_flow_boundary_package_result(result))
        print(f"\nFlow boundary-condition report written: {result.report_path}")
        return 0

    if args.command == "build-flow-1d-model":
        result = build_flow_1d_model(
            graph_yaml_path=args.graph,
            boundary_config_path=args.boundary_config,
            output_dir=args.output_dir,
            case_id=args.case_id,
            blood_viscosity_cp=args.blood_viscosity_cp,
            arterial_inlet_pressure_pa=args.arterial_inlet_pressure_pa,
            venous_outlet_pressure_pa=args.venous_outlet_pressure_pa,
            report_path=args.report,
        )
        print(format_flow_1d_model_result(result))
        print(f"\n1D flow model report written: {result.report_path}")
        return 0

    if args.command == "build-pulsatile-flow-model":
        result = build_pulsatile_flow_model(
            flow_1d_model_path=args.flow_1d_model,
            boundary_config_path=args.boundary_config,
            output_dir=args.output_dir,
            case_id=args.case_id,
            heart_rate_bpm=args.heart_rate_bpm,
            samples_per_cycle=args.samples_per_cycle,
            settling_cycles=args.settling_cycles,
            rcr_proximal_resistance_fraction=args.rcr_proximal_resistance_fraction,
            rcr_time_constant_s=args.rcr_time_constant_s,
            venous_pulsatility_fraction=args.venous_pulsatility_fraction,
            venous_phase_lag_fraction=args.venous_phase_lag_fraction,
            pressure_reference_weight=args.pressure_reference_weight,
            report_path=args.report,
        )
        print(format_pulsatile_flow_result(result))
        print(f"\nPulsatile flow model report written: {result.report_path}")
        return 0

    if args.command == "build-coupled-pulsatile-flow-model":
        result = build_coupled_pulsatile_flow_model(
            flow_1d_model_path=args.flow_1d_model,
            boundary_config_path=args.boundary_config,
            output_dir=args.output_dir,
            case_id=args.case_id,
            heart_rate_bpm=args.heart_rate_bpm,
            samples_per_cycle=args.samples_per_cycle,
            settling_cycles=args.settling_cycles,
            rcr_proximal_resistance_fraction=args.rcr_proximal_resistance_fraction,
            rcr_time_constant_s=args.rcr_time_constant_s,
            venous_pulsatility_fraction=args.venous_pulsatility_fraction,
            venous_phase_lag_fraction=args.venous_phase_lag_fraction,
            report_path=args.report,
        )
        print(format_coupled_pulsatile_flow_result(result))
        print(f"\nCoupled pulsatile flow model report written: {result.report_path}")
        return 0

    if args.command == "render-4d-flow":
        result = build_4d_flow_visualization(
            graph_yaml_path=args.graph,
            edge_timeseries_csv_path=args.edge_timeseries,
            node_timeseries_csv_path=args.node_timeseries,
            output_dir=args.output_dir,
            case_id=args.case_id,
            context_scene_spec_path=args.context_scene_spec,
            color_by=args.color_by,
            frame_count=args.frame_count,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
            zoom=args.zoom,
            context_group_ids=tuple(args.context_groups),
            max_context_triangles_per_group=args.max_context_triangles_per_group,
            label_boundary_nodes=args.label_boundary_nodes,
            gif_duration_ms=args.gif_duration_ms,
            report_path=args.report,
        )
        print(format_4d_flow_visualization_result(result))
        print(f"\n4D flow visualization report written: {result.report_path}")
        return 0

    if args.command == "build-radiotherapy-qa-package":
        result = build_radiotherapy_qa_package(
            combined_spec_path=args.combined_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            scenario=args.scenario,
            target_radius_mm=args.target_radius_mm,
            ptv_margin_mm=args.ptv_margin_mm,
            report_path=args.report,
        )
        print(format_radiotherapy_qa_package_result(result))
        print(f"\nRadiotherapy QA package report written: {result.report_path}")
        return 0

    if args.command == "build-rt-planning-bundle":
        result = build_rt_planning_bundle(
            rt_package_spec_path=args.rt_package_spec,
            coupled_flow_model_path=args.coupled_flow_model,
            output_dir=args.output_dir,
            case_id=args.case_id,
            prescription_dose_gy=args.prescription_dose_gy,
            vascular_dose_sensitivity=args.vascular_dose_sensitivity,
            export_dicom=args.export_dicom,
            report_path=args.report,
        )
        print(format_rt_planning_bundle_result(result))
        print(f"\nRT planning bundle report written: {result.report_path}")
        return 0

    if args.command == "analyze-spatial-rt-flow-coupling":
        result = analyze_spatial_rt_flow_coupling(
            rt_package_spec_path=args.rt_package_spec,
            rt_planning_spec_path=args.rt_planning_spec,
            vascular_graph_path=args.vascular_graph,
            edge_timeseries_csv_path=args.edge_timeseries,
            output_dir=args.output_dir,
            case_id=args.case_id,
            sample_step_mm=args.sample_step_mm,
            influence_radius_mm=args.influence_radius_mm,
            coordinate_mode=args.coordinate_mode,
            report_path=args.report,
        )
        print(format_spatial_rt_flow_coupling_result(result))
        print(f"\nSpatial RT-flow coupling report written: {result.report_path}")
        return 0

    if args.command == "build-spatial-rt-flow-dose-model":
        result = build_spatial_rt_flow_dose_model(
            rt_package_spec_path=args.rt_package_spec,
            rt_planning_spec_path=args.rt_planning_spec,
            vascular_graph_path=args.vascular_graph,
            edge_timeseries_csv_path=args.edge_timeseries,
            edge_coupling_csv_path=args.edge_coupling_csv,
            output_dir=args.output_dir,
            case_id=args.case_id,
            sample_step_mm=args.sample_step_mm,
            influence_falloff_mm=args.influence_falloff_mm,
            vascular_dose_sensitivity=args.vascular_dose_sensitivity,
            max_fractional_perturbation=args.max_fractional_perturbation,
            max_edges=args.max_edges,
            min_coupling_score=args.min_coupling_score,
            coordinate_mode=args.coordinate_mode,
            report_path=args.report,
        )
        print(format_spatial_rt_flow_dose_result(result))
        print(f"\nSpatial RT-flow dose model report written: {result.report_path}")
        return 0

    if args.command == "compare-scalar-vs-spatial-rt-flow-dose":
        result = compare_scalar_vs_spatial_rt_flow_dose(
            scalar_rt_planning_spec_path=args.scalar_rt_planning_spec,
            spatial_rt_flow_dose_spec_path=args.spatial_rt_flow_dose_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_rt_flow_model_comparison_result(result))
        print(f"\nScalar vs spatial RT-flow comparison report written: {result.report_path}")
        return 0

    if args.command == "build-current-phantom-status-atlas":
        result = build_current_phantom_status_atlas(
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_current_phantom_status_atlas_result(result))
        print(f"\nCurrent phantom status atlas report written: {result.report_path}")
        return 0

    if args.command == "build-corrected-branch-status-report":
        result = build_corrected_branch_status_report(
            output_dir=args.output_dir,
            case_id=args.case_id,
            vessel_flow_manifest_path=args.vessel_flow_manifest,
            vessel_flow_spec_path=args.vessel_flow_spec,
            vessel_flow_preview_path=args.vessel_flow_preview,
            rt_package_spec_path=args.rt_package_spec,
            rt_qa_preview_path=args.rt_qa_preview,
            coupled_flow_model_path=args.coupled_flow_model,
            coupled_flow_preview_path=args.coupled_flow_preview,
            flow4d_spec_path=args.flow4d_spec,
            spatial_coupling_spec_path=args.spatial_coupling_spec,
            spatial_coupling_csv_path=args.spatial_coupling_csv,
            spatial_coupling_preview_path=args.spatial_coupling_preview,
            spatial_dose_spec_path=args.spatial_dose_spec,
            spatial_dose_preview_path=args.spatial_dose_preview,
            gamma_spec_path=args.gamma_spec,
            gamma_qa_preview_path=args.gamma_qa_preview,
            report_path=args.report,
        )
        print(format_corrected_branch_status_result(result))
        print(f"\nCorrected branch-labelled status report written: {result.report_path}")
        return 0

    if args.command == "build-corrected-branch-release-package":
        result = build_corrected_branch_release_package(
            status_manifest_path=args.status_manifest,
            output_dir=args.output_dir,
            case_id=args.case_id,
            release_id=args.release_id,
            copy_small_artifacts=args.copy_small_artifacts,
            large_file_threshold_mb=args.large_file_threshold_mb,
            report_path=args.report,
        )
        print(format_corrected_branch_release_result(result))
        print(f"\nCorrected branch-labelled release report written: {result.report_path}")
        return 0

    if args.command == "audit-corrected-branch-release-package":
        result = audit_corrected_branch_release_package(
            release_manifest_path=args.release_manifest,
            output_dir=args.output_dir,
            audit_id=args.audit_id,
            report_path=args.report,
        )
        print(format_corrected_branch_release_audit_result(result))
        print(f"\nCorrected branch-labelled release audit report written: {result.report_path}")
        return 0

    if args.command == "build-research-release-package":
        result = build_research_release_package(
            stage_root=args.stage_root,
            reports_dir=args.reports_dir,
            output_dir=args.output_dir,
            case_id=args.case_id,
            release_id=args.release_id,
            report_path=args.report,
            copy_small_artifacts=args.copy_small_artifacts,
            large_file_threshold_mb=args.large_file_threshold_mb,
        )
        print(format_research_release_package_result(result))
        print(f"\nResearch release package report written: {result.report_path}")
        return 0

    if args.command == "audit-research-release-package":
        result = audit_research_release_package(
            release_manifest_path=args.release_manifest,
            output_dir=args.output_dir,
            audit_id=args.audit_id,
            report_path=args.report,
        )
        print(format_release_readiness_audit_result(result))
        print(f"\nRelease readiness audit report written: {result.report_path}")
        return 0

    if args.command == "promote-stage007-clean-baseline":
        result = promote_stage007_clean_baseline(
            release_manifest_path=args.release_manifest,
            stage_root=args.stage_root,
            case_id=args.case_id,
            baseline_id=args.baseline_id,
            graph_path=args.graph,
            voxelized_spec_path=args.voxelized_spec,
            release_archive_path=args.release_archive,
            report_path=args.report,
            write_accepted_aliases=args.write_accepted_aliases,
        )
        print(format_stage007_baseline_promotion_result(result))
        print(f"\nStage 007 active baseline promotion report written: {result.report_path}")
        return 0

    if args.command == "run-stage007-acceptance-smoke":
        result = run_stage007_acceptance_smoke(
            release_manifest_path=args.release_manifest,
            baseline_manifest_path=args.baseline_manifest,
            release_archive_path=args.release_archive,
            output_dir=args.output_dir,
            report_path=args.report,
            case_id=args.case_id,
            baseline_id=args.baseline_id,
            flow_mass_residual_threshold_ml_s=args.flow_mass_residual_threshold_ml_s,
            flow_split_review_threshold_pp=args.flow_split_review_threshold_pp,
            flow_split_fail_threshold_pp=args.flow_split_fail_threshold_pp,
            min_boundary_count=args.min_boundary_count,
        )
        print(format_stage007_acceptance_smoke_result(result))
        print(f"\nStage 007 acceptance smoke report written: {result.report_path}")
        return 0

    if args.command == "build-validation-roadmap":
        result = build_validation_roadmap(
            readiness_audit_yaml_path=args.readiness_audit,
            roadmap_csv_path=args.roadmap_csv,
            output_dir=args.output_dir,
            roadmap_id=args.roadmap_id,
            report_path=args.report,
        )
        print(format_validation_roadmap_result(result))
        print(f"\nValidation roadmap report written: {result.report_path}")
        return 0

    if args.command == "build-research-demonstrator-package":
        result = build_research_demonstrator_package(
            release_manifest_path=args.release_manifest,
            audit_yaml_path=args.audit_yaml,
            status_manifest_path=args.status_manifest,
            validation_roadmap_path=args.validation_roadmap,
            validation_intake_path=args.validation_intake,
            vessel_harmonization_path=args.vessel_harmonization,
            output_dir=args.output_dir,
            package_id=args.package_id,
            report_path=args.report,
        )
        print(format_research_demonstrator_result(result))
        print(f"\nResearch demonstrator report written: {result.report_path}")
        return 0

    if args.command == "build-validation-intake-package":
        result = build_validation_intake_package(
            cases_csv_path=args.cases_csv,
            output_dir=args.output_dir,
            intake_id=args.intake_id,
            required_vessel_labels=tuple(args.required_vessel_labels) if args.required_vessel_labels else None,
            report_path=args.report,
        )
        print(format_validation_intake_result(result))
        print(f"\nValidation intake report written: {result.report_path}")
        return 0

    if args.command == "discover-validation-candidates":
        result = discover_validation_candidates(
            search_roots=tuple(args.search_root or ("data", "outputs/digital/patient_input_adapter_stage001")),
            output_dir=args.output_dir,
            discovery_id=args.discovery_id,
            required_vessel_labels=tuple(args.required_vessel_labels) if args.required_vessel_labels else None,
            max_ct_org_cases=args.max_ct_org_cases,
            max_loose_nifti_cases=args.max_loose_nifti_cases,
            report_path=args.report,
        )
        print(format_validation_discovery_result(result))
        print(f"\nValidation candidate discovery report written: {result.report_path}")
        return 0

    if args.command == "stage-validation-case":
        result = stage_validation_case(
            case_id=args.case_id,
            source_dataset=args.source_dataset,
            ct_path=args.ct,
            cta_path=args.cta,
            ctv_path=args.ctv,
            organ_seg_path=args.organ_seg,
            vessel_seg_path=args.vessel_seg,
            vessel_label_config=args.vessel_label_config,
            output_dir=args.output_dir,
            required_vessel_labels=tuple(args.required_vessel_labels) if args.required_vessel_labels else None,
            access_status=args.access_status,
            notes=args.notes,
            copy_inputs=args.copy_inputs,
            report_path=args.report,
        )
        print(format_validation_case_staging_result(result))
        print(f"\nValidation case staging report written: {result.report_path}")
        return 0

    if args.command == "harmonize-vessel-labels":
        result = harmonize_vessel_labels(
            vessel_seg_path=args.vessel_seg,
            case_id=args.case_id,
            output_dir=args.output_dir,
            target_label_config=args.target_label_config,
            mapping_csv_path=args.mapping_csv,
            required_vessel_labels=tuple(args.required_vessel_labels) if args.required_vessel_labels else None,
            auto_identity=args.auto_identity,
            unmapped_policy=args.unmapped_policy,
            report_path=args.report,
        )
        print(format_vessel_label_harmonization_result(result))
        print(f"\nVessel label harmonization report written: {result.report_path}")
        return 0

    if args.command == "promote-harmonized-vessel-case":
        result = promote_harmonized_vessel_case(
            staged_case_manifest_path=args.staged_case_manifest,
            vessel_harmonization_manifest_path=args.vessel_harmonization_manifest,
            promoted_case_id=args.promoted_case_id,
            output_dir=args.output_dir,
            report_path=args.report,
        )
        print(format_validation_case_promotion_result(result))
        print(f"\nHarmonized vessel case-promotion report written: {result.report_path}")
        return 0

    if args.command == "resample-vessel-to-ct-grid":
        result = resample_vessel_to_ct_grid(
            ct_path=args.ct,
            vessel_seg_path=args.vessel_seg,
            staged_case_manifest_path=args.staged_case_manifest,
            target_mask_path=args.target_mask,
            case_id=args.case_id,
            output_dir=args.output_dir,
            alignment_mode=args.alignment_mode,
            required_vessel_labels=tuple(args.required_vessel_labels) if args.required_vessel_labels else None,
            report_path=args.report,
        )
        print(format_vessel_ct_grid_resample_result(result))
        print(f"\nCT-grid vessel resampling report written: {result.report_path}")
        return 0

    if args.command == "qa-vessel-label-anatomy":
        result = qa_vessel_label_anatomy(
            anatomy_labels_path=args.anatomy_labels,
            vessel_labels_path=args.vessel_labels,
            case_id=args.case_id,
            output_dir=args.output_dir,
            vessel_label_config=args.vessel_label_config,
            required_vessel_labels=tuple(args.required_vessel_labels) if args.required_vessel_labels else None,
            report_path=args.report,
        )
        print(format_vessel_label_anatomy_qa_result(result))
        print(f"\nVessel label anatomy QA report written: {result.report_path}")
        return 0

    if args.command == "correct-vessel-label-anatomy":
        result = correct_vessel_label_anatomy(
            anatomy_labels_path=args.anatomy_labels,
            vessel_labels_path=args.vessel_labels,
            case_id=args.case_id,
            output_dir=args.output_dir,
            vessel_label_config=args.vessel_label_config,
            max_regrow_iterations=args.max_regrow_iterations,
            report_path=args.report,
        )
        print(format_vessel_label_anatomy_correction_result(result))
        print(f"\nVessel label anatomy correction report written: {result.report_path}")
        return 0

    if args.command == "build-label-vessel-flow-domain":
        result = build_label_vessel_flow_domain(
            anatomy_labels_path=args.anatomy_labels,
            vessel_labels_path=args.vessel_labels,
            graph_yaml_path=args.graph,
            case_id=args.case_id,
            output_dir=args.output_dir,
            vessel_label_config=args.vessel_label_config,
            materials_path=args.materials,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            contrast_mode=args.contrast_mode,
            boundary_snap_radius_mm=args.boundary_snap_radius_mm,
            snap_boundary_nodes=args.snap_boundary_nodes,
            write_material_volumes=args.write_material_volumes,
            report_path=args.report,
        )
        print(format_label_vessel_flow_domain_result(result))
        print(f"\nLabelled vessel flow-domain report written: {result.report_path}")
        return 0

    if args.command == "build-dose-gamma-qa":
        result = build_dose_gamma_qa(
            pymedphys_eval_config_path=args.pymedphys_eval_config,
            output_dir=args.output_dir,
            case_id=args.case_id,
            dose_percent_threshold=args.dose_percent_threshold,
            distance_mm_threshold=args.distance_mm_threshold,
            lower_percent_dose_cutoff=args.lower_percent_dose_cutoff,
            interp_fraction=args.interp_fraction,
            max_gamma=args.max_gamma,
            local_gamma=args.local_gamma,
            random_subset=args.random_subset,
            random_seed=args.random_seed,
            write_volume_outputs=args.write_volume_outputs,
            report_path=args.report,
        )
        print(format_dose_gamma_qa_result(result))
        print(f"\nDose gamma QA report written: {result.report_path}")
        return 0

    if args.command == "build-printable-cartridge":
        result = build_printable_vascular_cartridge(
            labels_path=args.labels,
            flow_loop_spec_path=args.flow_loop_spec,
            output_dir=args.output_dir,
            label_id=args.label_id,
            centerline_csv_path=args.centerline_csv,
            formats=tuple(args.formats),
            voxel_size_mm=args.voxel_size_mm,
            wall_thickness_mm=args.wall_thickness_mm,
            bore_clearance_mm=args.bore_clearance_mm,
            pressure_tap_wall_mm=args.pressure_tap_wall_mm,
            target_max_faces=args.target_max_faces,
            report_path=args.report,
        )
        print(format_printable_cartridge_result(result))
        print(f"\nPrintable cartridge report written: {result.report_path}")
        if args.qa_report:
            qa_meshes = [
                path
                for path in (*result.cartridge_paths, *result.fluid_core_paths)
                if Path(path).suffix.lower() == ".stl"
            ]
            qa_results = analyze_meshes(qa_meshes)
            qa_output = write_mesh_qa_report(
                qa_results,
                args.qa_report,
                title="Printable Vascular Cartridge Mesh QA Report",
            )
            print(f"Printable cartridge QA report written: {qa_output}")
        return 0

    if args.command == "build-digital-torso":
        result = build_digital_torso_phantom(
            ct_path=args.ct,
            labels_path=args.labels,
            labelmap_path=args.labelmap,
            materials_path=args.materials,
            output_dir=args.output_dir,
            case_id=args.case_id,
            body_threshold_hu=args.body_threshold_hu,
            report_path=args.report,
        )
        print(format_digital_torso_result(result))
        print(f"\nDigital torso report written: {result.report_path}")
        return 0

    if args.command == "build-combined-digital-phantom":
        target_center = tuple(args.target_center_mm) if args.target_center_mm is not None else None
        result = build_combined_digital_phantom(
            torso_material_labels_path=args.torso_material_labels,
            torso_body_mask_path=args.torso_body_mask,
            source_ct_path=args.source_ct,
            vascular_labels_path=args.vascular_labels,
            flow_loop_spec_path=args.flow_loop_spec,
            materials_path=args.materials,
            output_dir=args.output_dir,
            case_id=args.case_id,
            vascular_label_id=args.vascular_label_id,
            target_center_mm=target_center,
            vascular_scale=args.vascular_scale,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            report_path=args.report,
        )
        print(format_combined_digital_phantom_result(result))
        print(f"\nCombined digital phantom report written: {result.report_path}")
        return 0

    if args.command == "build-anthropometric-torso-morph":
        result = build_anthropometric_torso_morph(
            combined_spec_path=args.combined_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            baseline_height_cm=args.baseline_height_cm,
            baseline_bmi=args.baseline_bmi,
            baseline_waist_cm=args.baseline_waist_cm,
            abdomen_center_fraction=args.abdomen_center_fraction,
            abdomen_width_fraction=args.abdomen_width_fraction,
            morph_mode=args.morph_mode,
            xy_padding_voxels=args.xy_padding_voxels,
            report_path=args.report,
        )
        print(format_anthropometric_torso_morph_result(result))
        print(f"\nAnthropometric morph report written: {result.report_path}")
        return 0

    if args.command == "build-statistical-anatomy-morph":
        result = build_statistical_anatomy_morph(
            combined_spec_path=args.combined_spec,
            population_label_paths=tuple(args.population_labels),
            output_dir=args.output_dir,
            case_id=args.case_id,
            population_case_ids=tuple(args.population_case_ids) if args.population_case_ids is not None else None,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            baseline_height_cm=args.baseline_height_cm,
            baseline_bmi=args.baseline_bmi,
            mode_weights=tuple(args.mode_weights),
            max_modes=args.max_modes,
            adipose_layer_mm=args.adipose_layer_mm,
            report_path=args.report,
        )
        print(format_statistical_anatomy_morph_result(result))
        print(f"\nStatistical anatomy morph report written: {result.report_path}")
        return 0

    if args.command == "build-population-cohort":
        result = build_population_cohort(
            combined_spec_path=args.combined_spec,
            population_label_paths=tuple(args.population_labels),
            output_dir=args.output_dir,
            cohort_id=args.cohort_id,
            population_case_ids=tuple(args.population_case_ids) if args.population_case_ids is not None else None,
            max_modes=args.max_modes,
            min_body_dice=args.min_body_dice,
            min_body_overlap=args.min_body_overlap,
            max_atlas_cases=args.max_atlas_cases,
            essential_group_ids=tuple(args.essential_groups),
            report_path=args.report,
        )
        print(format_population_cohort_result(result))
        print(f"\nPopulation cohort report written: {result.report_path}")
        return 0

    if args.command == "stage-ct-org-label-cohort":
        result = stage_ct_org_label_cohort(
            case_indices=tuple(args.case_indices),
            raw_label_dir=args.raw_label_dir,
            output_dir=args.output_dir,
            case_id_prefix=args.case_id_prefix,
            label_base_url=args.label_base_url,
            body_padding_mm=args.body_padding_mm,
            adipose_layer_mm=args.adipose_layer_mm,
            force_download=args.force_download,
            report_path=args.report,
        )
        print(format_ct_org_label_cohort_result(result))
        print(f"\nCT-ORG label cohort report written: {result.report_path}")
        return 0

    if args.command == "stage-avt-kits-aorta-cohort":
        result = stage_avt_kits_aorta_zip(
            zip_path=args.zip_path,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            case_ids=tuple(args.case_ids) if args.case_ids is not None else None,
            max_cases=args.max_cases,
            case_id_prefix=args.case_id_prefix,
            report_path=args.report,
        )
        print(format_avt_kits_aorta_result(result))
        print(f"\nAVT/KiTS aorta staging report written: {result.report_path}")
        return 0

    if args.command == "stage-reg-training-testing":
        result = stage_reg_training_testing_zip(
            zip_path=args.zip_path,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            target_case_ids=tuple(args.target_case_ids),
            max_targets=args.max_targets,
            max_pairs_per_target=args.max_pairs_per_target,
            extract_images=args.extract_images,
            report_path=args.report,
        )
        print(format_reg_training_testing_staging_result(result))
        print(f"\nReg-Training-Testing staging report written: {result.report_path}")
        return 0

    if args.command == "build-reg-training-testing-benchmark":
        result = build_reg_training_testing_benchmark(
            staged_manifest_path=args.staged_manifest,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            labelmap_path=args.labelmap,
            min_consensus_fraction=args.min_consensus_fraction,
            min_volume_weighted_dice=args.min_volume_weighted_dice,
            min_mean_dice=args.min_mean_dice,
            max_centroid_dispersion_mm=args.max_centroid_dispersion_mm,
            min_mean_ncc=args.min_mean_ncc,
            intensity_sample_stride=args.intensity_sample_stride,
            report_path=args.report,
        )
        print(format_reg_training_testing_benchmark_result(result))
        print(f"\nReg-Training-Testing benchmark report written: {result.report_path}")
        return 0

    if args.command == "rank-registration-anchors":
        thresholds = RegistrationAnchorThresholds(
            approve_min_mean_dice=args.approve_min_mean_dice,
            approve_min_target_mean_dice=args.approve_min_target_mean_dice,
            approve_max_mean_volume_cv=args.approve_max_mean_volume_cv,
            approve_max_target_centroid_dispersion_mm=args.approve_max_target_centroid_dispersion_mm,
            review_min_mean_dice=args.review_min_mean_dice,
            review_min_target_mean_dice=args.review_min_target_mean_dice,
            review_max_mean_volume_cv=args.review_max_mean_volume_cv,
            review_max_target_centroid_dispersion_mm=args.review_max_target_centroid_dispersion_mm,
        )
        result = rank_registration_anchors(
            benchmark_spec_path=args.benchmark_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
            thresholds=thresholds,
        )
        print(format_registration_anchor_qa_result(result))
        print(f"\nRegistration anchor QA report written: {result.report_path}")
        return 0

    if args.command == "run-deformation-match-experiment":
        result = run_deformation_match_experiment(
            experiment_id=args.experiment_id,
            output_dir=args.output_dir,
            ct_org_manifest_csv=args.ct_org_manifest,
            btcv_label_path=args.btcv_label,
            reg_training_manifest_path=args.reg_training_manifest,
            avt_aorta_manifest_csv=args.avt_aorta_manifest,
            profile_spec_glob=args.profile_spec_glob,
            include_metric_scaled_height_grid=args.include_height_grid,
            report_path=args.report,
        )
        print(format_deformation_match_experiment_result(result))
        print(f"\nDeformation match experiment report written: {result.report_path}")
        return 0

    if args.command == "generate-pca-mode-variants":
        result = generate_pca_mode_variants(
            combined_spec_path=args.combined_spec,
            cohort_spec_path=args.cohort_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            mode_count=args.mode_count,
            amplitude=args.amplitude,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            baseline_height_cm=args.baseline_height_cm,
            baseline_bmi=args.baseline_bmi,
            max_modes=args.max_modes,
            adipose_layer_mm=args.adipose_layer_mm,
            report_path=args.report,
        )
        print(format_pca_mode_variant_atlas_result(result))
        print(f"\nPCA mode variant atlas report written: {result.report_path}")
        return 0

    if args.command == "qa-pca-modes":
        result = rank_pca_modes(
            metrics_csv_path=args.metrics_csv,
            atlas_spec_path=args.atlas_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            max_waist_delta_cm=args.max_waist_delta_cm,
            max_body_delta_l=args.max_body_delta_l,
            max_group_delta_percent=args.max_group_delta_percent,
            max_vascular_volume_delta_percent=args.max_vascular_volume_delta_percent,
            expected_vascular_components=args.expected_vascular_components,
            min_score_for_approval=args.min_score_for_approval,
            report_path=args.report,
        )
        print(format_pca_mode_qa_result(result))
        print(f"\nPCA mode QA report written: {result.report_path}")
        return 0

    if args.command == "build-approved-pca-phantom-set":
        result = build_approved_pca_phantom_set(
            qa_decisions_path=args.qa_decisions,
            atlas_spec_path=args.atlas_spec,
            metrics_csv_path=args.metrics_csv,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
            max_preview_columns=args.max_preview_columns,
        )
        print(format_approved_pca_phantom_set_result(result))
        print(f"\nApproved PCA phantom-set report written: {result.report_path}")
        return 0

    if args.command == "build-user-profile-adapter":
        result = build_user_profile_adapter(
            approved_set_manifest_path=args.approved_set_manifest,
            metrics_csv_path=args.metrics_csv,
            output_dir=args.output_dir,
            profile_id=args.profile_id,
            case_id=args.case_id,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            baseline_height_cm=args.baseline_height_cm,
            baseline_bmi=args.baseline_bmi,
            waist_tolerance_cm=args.waist_tolerance_cm,
            body_volume_tolerance_l=args.body_volume_tolerance_l,
            warning_penalty=args.warning_penalty,
            report_path=args.report,
        )
        print(format_user_profile_adapter_result(result))
        print(f"\nUser profile adapter report written: {result.report_path}")
        return 0

    if args.command in {"build-patient-phantom-adapter", "build-patient-phantom"}:
        result = build_patient_phantom_adapter(
            input_ct_path=args.input_ct,
            input_cta_path=args.input_cta,
            input_ctv_path=args.input_ctv,
            organ_seg_path=args.organ_seg,
            gi_seg_path=args.gi_seg,
            vessel_seg_path=args.vessel_seg,
            patient_id=args.patient_id,
            case_id=args.case_id,
            adaptation_mode=args.adaptation_mode,
            output_dir=args.output_dir,
            organ_labelmap_path=args.organ_labelmap,
            gi_labelmap_path=args.gi_labelmap,
            materials_path=args.materials,
            approved_set_manifest_path=args.approved_set_manifest,
            baseline_graph_path=args.baseline_graph,
            baseline_combined_spec_path=args.baseline_combined_spec,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            copy_inputs=args.copy_inputs,
            report_path=args.report,
        )
        print(format_patient_phantom_adapter_result(result))
        print(f"\nPatient phantom input adapter report written: {result.report_path}")
        return 0

    if args.command == "stage-gi-segmentation":
        result = stage_gi_segmentation(
            ct_path=args.ct,
            gi_segmentation_path=args.gi_seg,
            gi_labelmap_path=args.gi_labelmap,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
            require_targets=tuple(args.require_targets),
        )
        print(format_gi_segmentation_staging_result(result))
        print(f"\nGI segmentation staging report written: {result.report_path}")
        return 0 if result.readiness_status in {"ready_for_real_gi_replacement", "partial_real_gi_replacement_ready"} else 1

    if args.command == "auto-stage-gi-segmentation":
        result = auto_stage_gi_segmentation(
            ct_path=args.ct,
            segmenter=args.segmenter,
            segmenter_output_dir=args.segmenter_output_dir,
            segmenter_executable=args.segmenter_executable,
            segmenter_args=args.segmenter_args,
            gi_labelmap_path=args.gi_labelmap,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
            require_targets=tuple(args.require_targets),
            force_rerun=args.force_rerun,
            dry_run=args.dry_run,
            timeout_s=args.timeout_s,
        )
        print(format_gi_autoseg_bridge_result(result))
        print(f"\nGI auto-segmentation bridge report written: {result.report_path}")
        return 0 if result.readiness_status in {"ready_for_real_gi_replacement", "partial_real_gi_replacement_ready", "planned_only"} else 1

    if args.command == "run-patient-phantom-build":
        result = run_patient_phantom_build(
            patient_manifest_path=args.patient_manifest,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
            organ_labelmap_path=args.organ_labelmap,
            materials_path=args.materials,
            baseline_graph_path=args.baseline_graph,
            allow_template_vessels=args.allow_template_vessels,
            dry_run=args.dry_run,
            run_rt=args.run_rt,
            export_dicom=args.export_dicom,
            sample_step_mm=args.sample_step_mm,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            arterial_inlet_flow_ml_s=args.arterial_inlet_flow_ml_s,
            heart_rate_bpm=args.heart_rate_bpm,
            organ_label_mode=args.organ_label_mode,
            correct_bone_conflicts=args.correct_bone_conflicts,
            bone_clearance_mm=args.bone_clearance_mm,
        )
        print(format_patient_phantom_build_result(result))
        print(f"\nPatient phantom build executor report written: {result.report_path}")
        return 1 if result.overall_status == "failed" else 0

    if args.command == "qa-patient-phantom-build":
        result = qa_patient_phantom_build(
            build_manifest_path=args.build_manifest,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
            max_organ_fail_edges=args.max_organ_fail_edges,
            max_organ_review_edges=args.max_organ_review_edges,
            max_radius_fail_edges=args.max_radius_fail_edges,
            max_radius_review_edges=args.max_radius_review_edges,
            expected_lumen_components=args.expected_lumen_components,
            max_overlap_after_cleanup=args.max_overlap_after_cleanup,
            max_outside_body_fraction=args.max_outside_body_fraction,
            max_flow_mass_residual_ml_s=args.max_flow_mass_residual_ml_s,
        )
        print(format_patient_build_qa_result(result))
        print(f"\nPatient phantom build QA gate report written: {result.report_path}")
        return 0

    if args.command in {"build-product-case", "build-case"}:
        result = build_product_case(
            input_ct_path=args.input_ct,
            input_cta_path=args.input_cta,
            input_ctv_path=args.input_ctv,
            organ_seg_path=args.organ_seg,
            gi_seg_path=args.gi_seg,
            vessel_seg_path=args.vessel_seg,
            existing_build_manifest_path=args.existing_build_manifest,
            patient_id=args.patient_id,
            case_id=args.case_id,
            output_dir=args.output_dir,
            organ_labelmap_path=args.organ_labelmap,
            gi_labelmap_path=args.gi_labelmap,
            materials_path=args.materials,
            baseline_graph_path=args.baseline_graph,
            baseline_combined_spec_path=args.baseline_combined_spec,
            approved_set_manifest_path=args.approved_set_manifest,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            copy_inputs=args.copy_inputs,
            allow_template_vessels=args.allow_template_vessels,
            dry_run=args.dry_run,
            run_rt=args.run_rt,
            export_dicom=args.export_dicom,
            sample_step_mm=args.sample_step_mm,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            arterial_inlet_flow_ml_s=args.arterial_inlet_flow_ml_s,
            heart_rate_bpm=args.heart_rate_bpm,
            organ_label_mode=args.organ_label_mode,
            correct_bone_conflicts=args.correct_bone_conflicts,
            bone_clearance_mm=args.bone_clearance_mm,
            run_qa=args.run_qa,
            qa_expected_lumen_components=args.qa_expected_lumen_components,
            render_3d=args.render_3d,
            existing_render_preview_path=args.existing_render_preview,
            existing_render_scene_spec_path=args.existing_render_scene,
            render_target_max_faces=args.render_target_max_faces,
            render_vessel_visible=args.render_vessel_visible,
            report_path=args.report,
        )
        print(format_product_case_result(result))
        print(f"\nProduct case report written: {result.report_path}")
        return 0

    if args.command in {"run-patient-case-adapter", "adapt-patient-case"}:
        result = run_patient_case_adapter(
            input_ct_path=args.input_ct,
            input_cta_path=args.input_cta,
            input_ctv_path=args.input_ctv,
            organ_seg_path=args.organ_seg,
            vessel_seg_path=args.vessel_seg,
            patient_id=args.patient_id,
            case_id=args.case_id,
            output_dir=args.output_dir,
            profile_spec_glob=args.profile_spec_glob,
            profile_metrics_csv_path=args.profile_metrics_csv,
            include_metric_scaled_height_grid=args.include_height_grid,
            organ_labelmap_path=args.organ_labelmap,
            materials_path=args.materials,
            baseline_graph_path=args.baseline_graph,
            baseline_combined_spec_path=args.baseline_combined_spec,
            approved_set_manifest_path=args.approved_set_manifest,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            copy_inputs=args.copy_inputs,
            allow_template_vessels=args.allow_template_vessels,
            dry_run=args.dry_run,
            run_rt=args.run_rt,
            export_dicom=args.export_dicom,
            sample_step_mm=args.sample_step_mm,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            arterial_inlet_flow_ml_s=args.arterial_inlet_flow_ml_s,
            heart_rate_bpm=args.heart_rate_bpm,
            organ_label_mode=args.organ_label_mode,
            correct_bone_conflicts=args.correct_bone_conflicts,
            bone_clearance_mm=args.bone_clearance_mm,
            run_qa=args.run_qa,
            qa_expected_lumen_components=args.qa_expected_lumen_components,
            render_3d=args.render_3d,
            render_target_max_faces=args.render_target_max_faces,
            score_only=args.score_only,
            report_path=args.report,
        )
        print(format_patient_case_adapter_result(result))
        print(f"\nPatient case adapter report written: {result.report_path}")
        return 0

    if args.command == "build-product-release-package":
        result = build_product_release_package(
            product_manifest_path=args.product_manifest,
            output_dir=args.output_dir,
            release_id=args.release_id,
            copy_small_artifacts=args.copy_small_artifacts,
            large_threshold_bytes=args.large_threshold_bytes,
            command_lines=tuple(args.command_lines),
            supplemental_artifact_paths=tuple(args.supplemental_artifacts),
            report_path=args.report,
        )
        print(format_product_release_result(result))
        print(f"\nProduct release report written: {result.report_path}")
        return 0

    if args.command == "build-product-release-case":
        result = build_product_release_case(
            input_ct_path=args.input_ct,
            input_cta_path=args.input_cta,
            input_ctv_path=args.input_ctv,
            organ_seg_path=args.organ_seg,
            vessel_seg_path=args.vessel_seg,
            existing_build_manifest_path=args.existing_build_manifest,
            patient_id=args.patient_id,
            case_id=args.case_id,
            output_dir=args.output_dir,
            product_output_dir=args.product_output_dir,
            release_output_dir=args.release_output_dir,
            release_id=args.release_id,
            organ_labelmap_path=args.organ_labelmap,
            materials_path=args.materials,
            baseline_graph_path=args.baseline_graph,
            baseline_combined_spec_path=args.baseline_combined_spec,
            approved_set_manifest_path=args.approved_set_manifest,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            copy_inputs=args.copy_inputs,
            allow_template_vessels=args.allow_template_vessels,
            dry_run=args.dry_run,
            run_rt=args.run_rt,
            export_dicom=args.export_dicom,
            sample_step_mm=args.sample_step_mm,
            vessel_wall_thickness_mm=args.vessel_wall_thickness_mm,
            arterial_inlet_flow_ml_s=args.arterial_inlet_flow_ml_s,
            heart_rate_bpm=args.heart_rate_bpm,
            organ_label_mode=args.organ_label_mode,
            correct_bone_conflicts=args.correct_bone_conflicts,
            bone_clearance_mm=args.bone_clearance_mm,
            run_qa=args.run_qa,
            qa_expected_lumen_components=args.qa_expected_lumen_components,
            render_3d=args.render_3d,
            existing_render_preview_path=args.existing_render_preview,
            existing_render_scene_spec_path=args.existing_render_scene,
            render_target_max_faces=args.render_target_max_faces,
            copy_small_release_artifacts=args.copy_small_release_artifacts,
            release_large_threshold_bytes=args.release_large_threshold_bytes,
            release_command_lines=tuple(args.release_command_lines),
            supplemental_artifact_paths=tuple(args.supplemental_artifacts),
            report_path=args.report,
        )
        print(format_product_release_case_result(result))
        print(f"\nProduct release case workflow report written: {result.workflow_report_path}")
        return 0

    if args.command == "build-profile-rerun-comparison-atlas":
        result = build_profile_rerun_comparison_atlas(
            output_dir=args.output_dir,
            case_id=args.case_id,
            profile_id=args.profile_id,
            profile_adapter_spec_path=args.profile_adapter_spec,
            anthropometric_spec_path=args.anthropometric_spec,
            reference_vascular_spec_path=args.reference_vascular_spec,
            profile_vascular_spec_path=args.profile_vascular_spec,
            reference_flow_spec_path=args.reference_flow_spec,
            profile_flow_spec_path=args.profile_flow_spec,
            reference_spatial_dose_spec_path=args.reference_spatial_dose_spec,
            profile_spatial_dose_spec_path=args.profile_spatial_dose_spec,
            reference_gamma_spec_path=args.reference_gamma_spec,
            profile_gamma_spec_path=args.profile_gamma_spec,
            report_path=args.report,
        )
        print(format_profile_rerun_comparison_result(result))
        print(f"\nProfile rerun comparison report written: {result.report_path}")
        return 0

    if args.command == "build-profile-sweep":
        result = build_profile_sweep(
            output_dir=args.output_dir,
            sweep_id=args.sweep_id,
            profile_specs=tuple(args.profile) if args.profile else None,
            combined_spec_path=args.combined_spec,
            baseline_graph_path=args.baseline_graph,
            baseline_labels_path=args.baseline_labels,
            materials_path=args.materials,
            baseline_height_cm=args.baseline_height_cm,
            baseline_bmi=args.baseline_bmi,
            arterial_inlet_flow_ml_s=args.arterial_inlet_flow_ml_s,
            export_dicom=args.export_dicom,
            gamma_random_subset=args.gamma_random_subset,
            high_bmi_waist_threshold_cm=args.high_bmi_waist_threshold_cm,
            high_bmi_xy_padding_voxels=args.high_bmi_xy_padding_voxels,
            padding_transition_margin_cm=args.padding_transition_margin_cm,
            report_path=args.report,
        )
        print(format_profile_sweep_result(result))
        print(f"\nProfile sweep report written: {result.report_path}")
        return 0

    if args.command == "build-profile-prescription":
        result = build_profile_operating_prescription(
            metrics_csv_path=args.metrics_csv,
            output_dir=args.output_dir,
            profile_id=args.profile_id,
            case_id=args.case_id,
            target_height_cm=args.target_height_cm,
            target_weight_kg=args.target_weight_kg,
            target_bmi=args.target_bmi,
            target_waist_cm=args.target_waist_cm,
            baseline_bmi=args.baseline_bmi,
            baseline_waist_cm=args.baseline_waist_cm,
            waist_tolerance_cm=args.waist_tolerance_cm,
            bmi_tolerance=args.bmi_tolerance,
            height_tolerance_cm=args.height_tolerance_cm,
            high_bmi_waist_threshold_cm=args.high_bmi_waist_threshold_cm,
            high_bmi_xy_padding_voxels=args.high_bmi_xy_padding_voxels,
            padding_transition_margin_cm=args.padding_transition_margin_cm,
            combined_spec_path=args.combined_spec,
            report_path=args.report,
        )
        print(format_profile_prescription_result(result))
        print(f"\nProfile prescription report written: {result.report_path}")
        return 0

    if args.command == "build-profile-envelope":
        result = build_profile_operating_envelope(
            metrics_csv_paths=tuple(args.metrics_csv),
            output_dir=args.output_dir,
            envelope_id=args.envelope_id,
            prescription_yaml_path=args.prescription_yaml,
            report_path=args.report,
        )
        print(format_profile_envelope_result(result))
        print(f"\nProfile envelope report written: {result.report_path}")
        return 0

    if args.command == "plan-next-profiles":
        result = plan_next_profile_validations(
            metrics_csv_path=args.metrics_csv,
            output_dir=args.output_dir,
            plan_id=args.plan_id,
            high_bmi_waist_threshold_cm=args.high_bmi_waist_threshold_cm,
            high_bmi_xy_padding_voxels=args.high_bmi_xy_padding_voxels,
            padding_transition_margin_cm=args.padding_transition_margin_cm,
            transition_margin_cm=args.transition_margin_cm,
            min_distance_from_existing_cm=args.min_distance_from_existing_cm,
            max_candidates=args.max_candidates,
            gamma_random_subset=args.gamma_random_subset,
            report_path=args.report,
        )
        print(format_profile_planning_result(result))
        print(f"\nProfile validation plan report written: {result.report_path}")
        return 0

    if args.command == "run-phantom-experiment-set":
        result = run_phantom_experiment_set(
            approved_set_manifest_path=args.approved_set_manifest,
            rt_planning_spec_path=args.rt_planning_spec,
            dose_gamma_spec_path=args.dose_gamma_spec,
            flow_model_spec_path=args.flow_model_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_phantom_experiment_set_result(result))
        print(f"\nApproved PCA phantom experiment-set report written: {result.report_path}")
        return 0

    if args.command == "build-variant-rerun-harness":
        result = build_variant_rerun_harness(
            approved_set_manifest_path=args.approved_set_manifest,
            variant_id=args.variant_id,
            baseline_combined_spec_path=args.baseline_combined_spec,
            flow_model_spec_path=args.flow_model_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            stage_material_maps=args.stage_material_maps,
            report_path=args.report,
        )
        print(format_variant_rerun_harness_result(result))
        print(f"\nVariant rerun harness report written: {result.report_path}")
        return 0

    if args.command == "compare-variant-rt-impact":
        result = compare_variant_rt_impact(
            baseline_rt_planning_spec_path=args.baseline_rt_planning_spec,
            variant_rt_planning_spec_path=args.variant_rt_planning_spec,
            baseline_gamma_spec_path=args.baseline_gamma_spec,
            variant_gamma_spec_path=args.variant_gamma_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            variant_id=args.variant_id,
            report_path=args.report,
        )
        print(format_variant_rt_comparison_result(result))
        print(f"\nVariant RT impact comparison report written: {result.report_path}")
        return 0

    if args.command == "render-combined-3d":
        result = generate_combined_3d_render(
            combined_labels_path=args.combined_labels,
            output_dir=args.output_dir,
            case_id=args.case_id,
            formats=tuple(args.formats),
            target_max_faces=args.target_max_faces,
            report_path=args.report,
        )
        print(format_render3d_result(result))
        print(f"\n3D render report written: {result.report_path}")
        if args.qa_report:
            qa_meshes = [
                path
                for mesh in result.meshes
                for path in mesh.output_paths
                if Path(path).suffix.lower() == ".stl"
            ]
            qa_results = analyze_meshes(qa_meshes)
            qa_output = write_mesh_qa_report(
                qa_results,
                args.qa_report,
                title="Combined 3D Render Mesh QA Report",
            )
            print(f"3D render mesh QA report written: {qa_output}")
        return 0

    if args.command == "render-vascular-network-3d":
        result = generate_vascular_network_3d_render(
            context_labels_path=args.context_labels,
            arterial_lumen_mask_path=args.arterial_mask,
            venous_lumen_mask_path=args.venous_mask,
            flow_domain_labels_path=args.flow_domain_labels,
            vessel_wall_mask_path=args.vessel_wall_mask,
            output_dir=args.output_dir,
            case_id=args.case_id,
            formats=tuple(args.formats),
            target_max_faces=args.target_max_faces,
            vascular_graph_path=args.vascular_graph,
            render_vessel_visible_preview=args.render_vessel_visible,
            vessel_display_scale=args.vessel_display_scale,
            report_path=args.report,
        )
        print(format_render3d_result(result))
        print(f"\nVascular network 3D render report written: {result.report_path}")
        if args.qa_report:
            qa_meshes = [
                path
                for mesh in result.meshes
                for path in mesh.output_paths
                if Path(path).suffix.lower() == ".stl"
            ]
            qa_results = analyze_meshes(qa_meshes)
            qa_output = write_mesh_qa_report(
                qa_results,
                args.qa_report,
                title="Cleaned Vascular Network 3D Render Mesh QA Report",
            )
            print(f"Vascular network 3D render mesh QA report written: {qa_output}")
        return 0

    if args.command == "render-3d-atlas":
        result = generate_3d_view_atlas(
            scene_spec_path=args.scene_spec,
            output_dir=args.output_dir,
            case_id=args.case_id,
            report_path=args.report,
        )
        print(format_render_atlas_result(result))
        print(f"\n3D render atlas report written: {result.report_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
