from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class FieldDatasetSpec:
    repo_id: str
    priority: int
    recommended_stage: str
    task_family: str
    tags: Tuple[str, ...]
    authority: str
    source_url: str
    access_pattern: str
    requires_token: bool
    notes: str


FIELD_DATASETS: Dict[str, FieldDatasetSpec] = {
    "weatherbench2/era5-64x32": FieldDatasetSpec(
        repo_id="weatherbench2/era5-64x32",
        priority=1,
        recommended_stage="first_authoritative_real_field_run",
        task_family="global_weather_rollout",
        tags=("weather", "era5", "zarr", "time-series", "real-field", "authoritative"),
        authority="WeatherBench 2 / Google Research / ECMWF ERA5",
        source_url="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
        access_pattern="xarray.open_zarr via gcsfs/fsspec, then deterministic variable/time subset",
        requires_token=False,
        notes=(
            "Best first mature real-field benchmark. Use the low-resolution 64x32 ERA5 Zarr for audit, "
            "one-step, and short rollout before larger WeatherBench 2 resolutions."
        ),
    ),
    "weatherbench2/era5-240x121": FieldDatasetSpec(
        repo_id="weatherbench2/era5-240x121",
        priority=2,
        recommended_stage="official_resolution_scaling",
        task_family="global_weather_rollout",
        tags=(
            "weather",
            "era5",
            "zarr",
            "time-series",
            "real-field",
            "authoritative",
            "official-resolution",
        ),
        authority="WeatherBench 2 / Google Research / ECMWF ERA5",
        source_url=(
            "gs://weatherbench2/datasets/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
        ),
        access_pattern=(
            "xarray.open_zarr via gcsfs/fsspec. Current WCA cache path may project this to a square "
            "training grid; native WeatherBench evaluation requires rectangular lat/lon support."
        ),
        requires_token=False,
        notes=(
            "WeatherBench 2 paper evaluation resolution. Use after strict 64x32 matched baselines complete; "
            "do not claim official WeatherBench scores from square-resized caches."
        ),
    ),
    "weatherbench2/era5-wb13-1440x721": FieldDatasetSpec(
        repo_id="weatherbench2/era5-wb13-1440x721",
        priority=3,
        recommended_stage="full_weatherbench_pretraining",
        task_family="global_weather_rollout",
        tags=(
            "weather",
            "era5",
            "zarr",
            "time-series",
            "real-field",
            "authoritative",
            "full-resolution",
            "wb13",
        ),
        authority="WeatherBench 2 / Google Research / ECMWF ERA5",
        source_url=(
            "gs://weatherbench2/datasets/era5/"
            "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        ),
        access_pattern=(
            "streamed xarray/zarr with chunked time windows; requires native rectangular patching, "
            "latitude weighting, and staged caching before model training."
        ),
        requires_token=False,
        notes=(
            "High-resolution WeatherBench 2 ERA5 source with 13 pressure levels and derived variables. "
            "This is future pretraining/evaluation infrastructure, not the current compact experiment path."
        ),
    ),
    "weatherbenchx/evaluation": FieldDatasetSpec(
        repo_id="weatherbenchx/evaluation",
        priority=4,
        recommended_stage="official_scorecard_evaluation",
        task_family="weather_evaluation",
        tags=("weather", "weatherbench-x", "evaluation", "xarray", "beam", "scorecard"),
        authority="WeatherBench-X / Google Research",
        source_url="https://github.com/google-research/weatherbenchX",
        access_pattern=(
            "WeatherBench-X xarray data loaders, deterministic/probabilistic metrics, aggregation, "
            "and optional Apache Beam/Dataflow execution for official-style scorecards."
        ),
        requires_token=False,
        notes=(
            "Use for official-style WeatherBench evaluation after WCA can emit forecast datasets with "
            "init_time/lead_time/latitude/longitude/variable schema."
        ),
    ),
    "pdebench/darus-2986": FieldDatasetSpec(
        repo_id="pdebench/darus-2986",
        priority=5,
        recommended_stage="official_pdebench_strict_eval",
        task_family="pde_forward_prediction",
        tags=("pde", "scientific-ml", "hdf5", "darus", "authoritative"),
        authority="PDEBench / DaRUS / NeurIPS 2022",
        source_url="https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986",
        access_pattern=(
            "Use PDEBench data_download scripts or a manually uploaded DaRUS HDF5 file, then register "
            "source_manifest.json, dataset_audit.json, and cache_manifest.json before training."
        ),
        requires_token=False,
        notes=(
            "Primary V20 PDE benchmark authority. Community PDE datasets remain loader smoke tests until "
            "they are tied to a comparable source manifest and matched baselines."
        ),
    ),
    "google/arco-era5": FieldDatasetSpec(
        repo_id="google/arco-era5",
        priority=6,
        recommended_stage="large_authoritative_weather_scaling",
        task_family="global_weather_rollout",
        tags=("weather", "era5", "zarr", "time-series", "real-field", "authoritative", "large-scale"),
        authority="Google Cloud Public Dataset Program / ECMWF ERA5",
        source_url="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        access_pattern="xarray.open_zarr via gcsfs/fsspec; start with small time/variable/region slices",
        requires_token=False,
        notes=(
            "Authoritative large ARCO ERA5 source. Use after WeatherBench 2 64x32 works because the raw "
            "spatial and variable scale can dominate early model experiments."
        ),
    ),
    "mpi-sintel/optical-flow": FieldDatasetSpec(
        repo_id="mpi-sintel/optical-flow",
        priority=7,
        recommended_stage="visual_field_rollout_benchmark",
        task_family="optical_flow",
        tags=("video", "optical-flow", "synthetic-realistic", "benchmark", "mature"),
        authority="MPI Sintel optical flow benchmark",
        source_url="http://sintel.is.tue.mpg.de/",
        access_pattern="official archive download; convert frames/flow/masks into local audited cache",
        requires_token=False,
        notes=(
            "Mature optical-flow benchmark with long sequences and difficult visual motion. Good bridge from "
            "PDE/weather fields to visual dynamics, but less direct than ERA5 for first field rollout."
        ),
    ),
    "kitti/flow-2012": FieldDatasetSpec(
        repo_id="kitti/flow-2012",
        priority=8,
        recommended_stage="real_motion_probe",
        task_family="optical_flow",
        tags=("video", "optical-flow", "driving", "benchmark", "mature"),
        authority="KITTI Vision Benchmark Suite",
        source_url="http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow",
        access_pattern="official KITTI download or audited Hugging Face mirror; convert pairs/flow/calib locally",
        requires_token=False,
        notes=(
            "Highly used real-world motion benchmark. Best for later vision-field validation because sparse flow, "
            "occlusion, and camera calibration complicate the first loader."
        ),
    ),
    "sogeeking/vorticity": FieldDatasetSpec(
        repo_id="sogeeking/vorticity",
        priority=9,
        recommended_stage="loader_smoke_not_primary_evidence",
        task_family="2d_field_rollout",
        tags=("pde", "navier-stokes", "vorticity", "time-series", "hf-smoke"),
        authority="Hugging Face community PDE dataset",
        source_url="https://huggingface.co/datasets/sogeeking/vorticity",
        access_pattern="huggingface_hub snapshot_download; inspect HDF5 keys before training",
        requires_token=False,
        notes=(
            "Still useful as a compact HDF5 loader smoke test, but not the primary scientific benchmark because "
            "it is less mature and lower-usage than WeatherBench/ERA5, Sintel, or KITTI."
        ),
    ),
    "ashiq24/FSI-pde-dataset": FieldDatasetSpec(
        repo_id="ashiq24/FSI-pde-dataset",
        priority=10,
        recommended_stage="multi_physics_transfer",
        task_family="multi_physics_rollout",
        tags=("pde", "fluid-solid-interaction", "multi-physics"),
        authority="Hugging Face community multiphysics PDE dataset",
        source_url="https://huggingface.co/datasets/ashiq24/FSI-pde-dataset",
        access_pattern="huggingface_hub snapshot_download; convert simulation files after schema audit",
        requires_token=False,
        notes="Fluid-solid interaction simulations; useful after scalar field rollout works.",
    ),
    "AISDL-SNU/LithoBench-PDE": FieldDatasetSpec(
        repo_id="AISDL-SNU/LithoBench-PDE",
        priority=11,
        recommended_stage="image_to_field",
        task_family="image_to_image_pde",
        tags=("pde", "image-to-image", "lithography"),
        authority="LithoBench-PDE benchmark",
        source_url="https://huggingface.co/datasets/AISDL-SNU/LithoBench-PDE",
        access_pattern="Hugging Face dataset files; audit license and task mapping before use",
        requires_token=False,
        notes="High-fidelity photolithography PDE fields; good image-to-field benchmark.",
    ),
    "galilai-group/kitti-flow2012": FieldDatasetSpec(
        repo_id="galilai-group/kitti-flow2012",
        priority=12,
        recommended_stage="hf_mirror_audit_only",
        task_family="optical_flow",
        tags=("video", "optical-flow", "driving", "hf-mirror"),
        authority="Hugging Face mirror of KITTI Flow 2012",
        source_url="https://huggingface.co/datasets/galilai-group/kitti-flow2012",
        access_pattern="Use only after verifying it matches official KITTI files and labels",
        requires_token=False,
        notes="Convenient mirror candidate, not preferred over the official KITTI source for evidence.",
    ),
    "mteb/ucf101": FieldDatasetSpec(
        repo_id="mteb/ucf101",
        priority=13,
        recommended_stage="video_pretraining_probe",
        task_family="video_frame_prediction",
        tags=("video", "action", "frame-sequence"),
        authority="UCF101 mirror on Hugging Face",
        source_url="https://huggingface.co/datasets/mteb/ucf101",
        access_pattern="Use only after frame-continuity and split audit",
        requires_token=False,
        notes="Frame-extracted UCF101; useful later, but less physically controlled than PDE data.",
    ),
}


def get_field_dataset_spec(repo_id: str) -> FieldDatasetSpec:
    try:
        return FIELD_DATASETS[repo_id]
    except KeyError as exc:
        known = ", ".join(sorted(FIELD_DATASETS))
        raise KeyError(f"Unknown field dataset {repo_id!r}. Known datasets: {known}") from exc
