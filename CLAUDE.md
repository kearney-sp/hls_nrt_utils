# CLAUDE.md

## Project
`hlsstack` — pip-installable library for near-real-time (NRT) vegetation monitoring in rangelands from NASA HLS (Harmonized Landsat-Sentinel) imagery. Queries LPCLOUD STAC, builds lazy Dask/xarray stacks via `stackstac`, computes spectral indices, smooths time series, and runs trained ML models (biomass, cover, crude protein) on the stack. No CLI, no test suite, no CI config — treat it as a pure compute library consumed by external driver scripts.

## Build & run
Conda-managed, no poetry/uv/venv. Python 3.10.

```bash
# create/update env (pinned; includes GDAL/rasterio/rioxarray/dask-jobqueue stack)
conda env create -f hls_nrt_environment.yml
conda activate hls_nrt_env

# install this library into the env (editable, for development)
pip install -e .

# or, as a consumer would install it
pip install git+https://github.com/kearney-sp/hls_nrt_utils.git
```
No `requirements.txt`, `pyproject.toml`, or tests exist — don't assume `pytest`/CI hooks are present.

### HPC batch submission
There is no static `sbatch`/`qsub` script in this repo. Submission is programmatic via `hlsstack.utils.hpc_setup.launch_dask()`, which wraps `dask_jobqueue.SLURMCluster` and submits SLURM jobs itself:

```python
from hlsstack.utils.hpc_setup import launch_dask

client = launch_dask(
    cluster_loc='hpc',        # 'local' -> single-node LocalCluster (dev/small bbox only)
    hls=True,                 # also calls hls_funcs.fetch.setup_env() for GDAL/AWS env vars
    num_processes=1,
    num_threads_per_processes=2,
    mem_gb_per=2.5,            # GB per THREAD -> job mem = mem_gb_per * processes * threads
    num_jobs=16,               # max SLURM jobs; cluster adapts 0..num_jobs*2
    partition='scavenger',
    duration='02:00:00',       # walltime; workers self-recycle via --lifetime 2h/--lifetime-stagger 4m
    debug=False,               # True -> debug/slurm-%j.{out,err}; False -> /dev/null
)
```
If you wrap a driver script in an `sbatch` file, keep that outer job small (login/driver-node footprint only) — `launch_dask` fans out the real per-tile compute as its own separate SLURM jobs.

## Architecture boundaries — module map
Four top-level packages under `hlsstack/`: `hls_funcs/`, `models/`, `utils/`, `objects/`. Internal import edges (verified via grep, not assumed):

```
hls_funcs/bands.py    (leaf: numpy/xarray only)
hls_funcs/indices.py  (leaf: numpy only)
hls_funcs/mmodel.py   (leaf: numpy + joblib only; vendored mmodel_sel weight kernel,
                       bundle loader, embedding->cosine helper for pred_bm_mmodel)
hls_funcs/fetch.py    (leaf: STAC query + lazy stackstac build; no internal deps)
hls_funcs/smooth.py, smooth_new.py   (leaf: temporal smoothing; no internal deps)
models/load.py        (leaf: model_dict registry + load_model())
utils/convert.py      (leaf: KMZ/shapefile conversion)
objects/charts.py     (leaf: static ECharts dicts; fully decoupled, no imports of/from anything else)

hls_funcs/predict.py  --> hls_funcs.bands, hls_funcs.indices   (top-level `import *`, for func_dict)
                      --> hls_funcs.mmodel                      (top-level, for pred_bm_mmodel)
                      --> rioxarray                             (top-level, .rio accessor for pred_bm_mmodel)
                      --> models.load.load_model                (deferred import, pred_cp_old only)

hls_funcs/masks.py    --> hls_funcs.bands            (deferred import, in bolton_mask_xr)
                      --> utils.atsa_utils            (deferred import, in atsa_mask)
utils/atsa_utils.py   --> hls_funcs.masks.mask_hls    (top-level import)
   ^^ masks.py <-> atsa_utils.py is a circular pair, broken by deferring the masks.py side
      of the import inside the function body. Do NOT hoist that import to module level.

utils/hpc_setup.py    --> hls_funcs.fetch.setup_env   (deferred import, conditional on hls=True)
   ^^ utils/ is not strictly "below" hls_funcs/ — it reaches back in; treat utils/ and
      hls_funcs/ as peers, not a strict dependency layer.
```
Practical rule when adding code: `bands.py`/`indices.py` must stay leaves (no imports from `masks`/`predict`/`utils`) since they're imported both directly and via `predict.func_dict` — adding a cycle there breaks more than `masks`/`atsa_utils` does.

## Key libraries & conventions
- **xarray/Dask is the only object model** for imagery — drop to raw `numpy` only inside `apply_ufunc`/`map_blocks` kernels, convert back to `xr` immediately after.
- **rasterio/rioxarray**: CRS/reprojection semantics are inherited from `stackstac.stack(epsg=proj_epsg, ...)`; `masks.shp2mask` uses `rasterio.features.rasterize` directly against an `xr_object`'s coords — keep raster/vector alignment (`transform`, `outshape`) derived from the xarray object, not hardcoded.
- **fiona/geopandas** (`utils/convert.py`): must explicitly enable `fiona.drvsupport.supported_drivers['LIBKML'/'libkml'] = 'rw'` before reading `.kml`/`.kmz` — the driver is off by default.
- HLS nodata sentinel is `-9999`; every band/index function masks with `.where(band != -9999)` before deriving anything — keep this convention, don't introduce a different sentinel.
- `predict.pred_cp` requires `time` as a single chunk (`dat.chunk({'time': -1})`); it raises `ValueError` otherwise. Every other prediction path (`pred_bm`, `pred_cov`, `pred_bm_se`) tolerates `time: 1`.
- Model-loading convention: current code (`pred_bm`, `pred_cov`, `pred_cp`) takes an already-loaded `model` object as a parameter and closure-captures it into `apply_ufunc`/`map_blocks`. Only the legacy `pred_cp_old` reloads the model per-call via a deferred `models.load.load_model` import (worked around historical cross-process deserialization errors) — don't copy that pattern into new code; load once and pass the model in.
- Model registry: add new `.pk`/`.pkl` models to `models/load.py::model_dict` only — never hardcode a model path elsewhere in the codebase.
- `pred_bm_mmodel` is the odd one out: its `'mmodel_biomass'` registry entry is a plain dict of numpy arrays (no sklearn estimator, no `feature_names_in_`), and it also ships a package-data raster `models/mmodel_similarity_conus_2km.tif`. Both are regenerated from the `mmodel_sel` project — the bundle by its `scripts/i_export_mmodel_bundle.py`, the raster by its `scripts/b_embeddings.py --steps export --stack`. Don't point other prediction paths at `'mmodel_biomass'`.

## State management & logging
- No `logging` module anywhere in the library except one branch in `hpc_setup.py` (debug mode); all status/progress is `print()`. Match this — don't introduce a logging framework into `hlsstack`.
- HPC debug artifacts land in `debug/slurm-%j.{out,err}` only when `debug=True`; otherwise scheduler stdout/stderr goes to `/dev/null` — don't rely on SLURM logs unless `debug=True` is set.
- Prediction kernels (`pred_bm`, `pred_cp`, `pred_cov`) explicitly `del` intermediate arrays/DataFrames inside per-timestep loops to bound peak memory per Dask task. Preserve this when editing — removing the `del`s reintroduces OOM regressions previously fixed in `pred_cp` (see git history).

## Code health: chunking, memory, and worker boundaries
- Default `stackstac` chunksize is `(3660, 3660)` px; `fetch.build_xr` then rechunks to one chunk per time step (`y: -1, x: -1`). Widening spatial chunks trades worker count for per-task memory — validate on `cluster_loc='local'` with a small bbox before scaling to `hpc`.
- `mem_gb_per` in `launch_dask` is **per thread**, not per job — total per-job memory request is `mem_gb_per * num_processes * num_threads_per_processes`. Undersizing kills workers mid-tile; oversizing wastes shared scavenger-partition allocation.
- Keep `num_processes * num_threads_per_processes` modest per job on shared/scavenger partitions; scale throughput via `num_jobs` / `clust.adapt` (adaptive scaling), not thread count per worker.
- Worker `--lifetime 2h --lifetime-stagger 4m` recycling is intentional (bounds Dask memory creep on long HPC runs) — keep `worker_args` when customizing `launch_dask` calls.
- `smooth_new.smooth_array_parallel`/`smooth_xr_parallel` batch by `chunk_size` (pixels per joblib task, default 500–2000) — tune this down on memory-constrained shared nodes, not `n_jobs` alone. This path is local-process (`joblib`) parallelism only — it forces a full compute via `.values` and cannot span multiple SLURM nodes; use it for `cluster_loc='local'` dev/validation.
- `smooth_new.smooth_xr_dask` is the multi-node path — it distributes via `xr.apply_ufunc(dask='parallelized')` over whatever spatial (`y`, `x`) chunks the input already has, so it needs a Dask distributed `Client` to be active (else it silently falls back to Dask's local scheduler) and the core dim (`time`) as a single chunk, same as `pred_cp` above. Keep the input lazy — don't `.compute()`/`.values` it before calling this, or the whole array collapses onto whichever process calls it, same failure mode as `smooth_xr_parallel`.
