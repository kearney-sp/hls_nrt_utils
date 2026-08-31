import pickle
import os
import glob
import functools
import pandas as pd
import xarray as xr
import numpy as np
import random
import rioxarray  # noqa: F401 -- registers the .rio accessor used by pred_bm_mmodel
from pkg_resources import resource_filename
from hlsstack.hls_funcs.bands import *
from hlsstack.hls_funcs.indices import *
from hlsstack.hls_funcs import mmodel as _mm
from pysptools.abundance_maps import amaps
import scipy.stats as st
from sklearn.cross_decomposition import PLSRegression
import dask
import warnings
#from sklearn.exceptions import InconsistentVersionWarning

func_dict = {
    "BLUE": blue_func,
    "GREEN": green_func,
    "RED": red_func,
    "NIR1": nir_func,
    "SWIR1": swir1_func,
    "SWIR2": swir2_func,
    "NDVI": ndvi_func,
    "DFI": dfi_func,
    "NDTI": ndti_func,
    "SATVI": satvi_func,
    "NDII7": ndii7_func,
    'SAVI': savi_func,
    'RDVI': rdvi_func,
    'MTVI1': mtvi1_func,
    'NCI': nci_func,
    'NDCI': ndci_func,
    'PSRI': psri_func,
    'NDWI': ndwi_func,
    'EVI': evi_func,
    'TCBI': tcbi_func,
    'TCGI': tcgi_func,
    'TCWI': tcwi_func,
    "BAI_126": bai_126_func,
    "BAI_136": bai_136_func,
    "BAI_146": bai_146_func,
    "BAI_236": bai_236_func,
    "BAI_246": bai_246_func,
    "BAI_346": bai_346_func
}


def pred_bm(dat, model):
    model_vars = list(model.feature_names_in_)

    def pred_func(*args):
        # args each arrive as (time, y, x). Flatten y/x to a single pixel axis
        # here, with numpy, rather than upstream with xarray's
        # .stack(z=('y','x')) -- see pred_func_xr below for why.
        grid = args[0].shape[1:]
        mat = np.stack([a.reshape(a.shape[0], -1) for a in args], axis=-1).astype(np.float32)
        time_steps, n_pixels, n_bands = mat.shape
        max_f32 = np.finfo(np.float32).max

        out = np.full((time_steps, n_pixels), np.nan, dtype=np.float32)

        for t in range(time_steps):
            mat_t = mat[t]
            mat_t = np.where(np.isfinite(mat_t) & (np.abs(mat_t) <= max_f32), mat_t, np.nan)

            # Restore DataFrame to satisfy StandardScaler's feature name expectation
            df_t = pd.DataFrame(mat_t, columns=model_vars)
            valid_mask = ~df_t.isna().any(axis=1).values

            if valid_mask.any():
                preds = model.predict(df_t[valid_mask])
                out[t, valid_mask] = preds.squeeze()
                del preds

            del mat_t, df_t, valid_mask

        del mat
        return out.reshape((time_steps,) + grid)

    def pred_func_xr(dat_xr, model_vars_xr):
        # The index functions run on the native (time, y, x) grid. They used to
        # run on dat_xr.stack(z=('y','x')), which builds a pandas MultiIndex
        # with one entry per pixel -- and every elementwise op inside func_dict
        # then pays index alignment proportional to it. That was 92% of this
        # function's runtime: measured on one tbng-sized date (3000x2600, 7.8
        # Mpx), the 14 index functions took 25.3s stacked versus 0.76s
        # unstacked, against 0.06s for model.predict itself. Dropping the z
        # index alone (same stacked shape, no MultiIndex) also gives 0.61s, so
        # it is the index, not the reshape. pred_func flattens y/x with numpy
        # instead, where it is a view. Results are bit-identical.
        vars_list_xr = [func_dict[v](dat_xr) for v in model_vars_xr]

        return xr.apply_ufunc(
            pred_func,
            *vars_list_xr,
            dask='parallelized',
            vectorize=False,
            input_core_dims=[['time', 'y', 'x']] * len(model_vars_xr),
            output_core_dims=[['time', 'y', 'x']],
            output_dtypes=['int16']
        )

    return pred_func_xr(dat, model_vars)


# ------------------------------------------------------------------------------
# pred_bm_mmodel -- the mmodel_sel embedding-p=2 five-model biomass blend
# ------------------------------------------------------------------------------

#: default bundled similarity raster (6 bands: 5 model cosines + domain), written
#: by mmodel_sel `scripts/b_embeddings.py --steps export --stack` and shipped as
#: package data. int16, cosine * 1e4, nodata -32768, EPSG:5070.
_MMODEL_SIMILARITY_COG = 'models/mmodel_sel_similarity_stack_2000m.tif'
_MMODEL_SIM_SCALE = 1e4
_MMODEL_SIM_NODATA = -32768

#: reprojected similarity is cached per (bundle, output grid) so a `map_blocks`
#: run does not reproject the CONUS raster once per timestep.
_mmodel_sim_cache = {}


def _infer_crs(dat):
    try:
        if dat.rio.crs is not None:
            return dat.rio.crs
    except Exception:
        pass
    for key in ('crs', 'proj:epsg', 'epsg'):
        if key in dat.attrs:
            return dat.attrs[key]
    if 'epsg' in dat.coords:
        return 'EPSG:' + str(int(dat['epsg'].values))
    raise ValueError(
        'could not infer a CRS from `dat`; pass `similarity=` as a DataArray '
        'already on `dat`\'s grid, or write a CRS onto `dat` first'
    )


def _grid_key(ref):
    y, x = ref['y'].values, ref['x'].values
    return (str(ref.rio.crs), float(y[0]), float(y[-1]), y.size,
            float(x[0]), float(x[-1]), x.size)


def _prepare_similarity(dat, bundle, similarity, embedding):
    """Return a ``(model, y, x)`` DataArray of cosine bands + a `domain` band,
    aligned to ``dat``'s grid. ``model`` coord is ``bundle['model_keys'] + ['domain']``."""
    keys = list(bundle['model_keys'])
    band_names = keys + ['domain']

    ref = dat['NIR1'].isel(time=0) if 'time' in dat['NIR1'].dims else dat['NIR1']
    ref = ref.rio.write_crs(_infer_crs(dat))

    # already-aligned DataArray: trust it, just select this block's window
    if isinstance(similarity, xr.DataArray) and 'model' in similarity.dims:
        return similarity.sel(y=dat['y'], x=dat['x'], method='nearest')

    cache_key = (id(bundle), _grid_key(ref),
                 None if embedding is not None else str(similarity))
    if embedding is None and cache_key in _mmodel_sim_cache:
        return _mmodel_sim_cache[cache_key]

    if embedding is not None:
        emb = embedding
        if not isinstance(emb, xr.DataArray):
            emb = rioxarray.open_rasterio(emb)
        emb = emb.rio.write_crs(emb.rio.crs or _infer_crs(dat))
        emb = emb.rio.reproject_match(ref)
        arr = _mm.cosine_from_embedding(emb.values, bundle)
        sim = xr.DataArray(
            arr, dims=('model', 'y', 'x'),
            coords={'model': band_names, 'y': ref['y'], 'x': ref['x']},
        )
    else:
        path = similarity
        if path is None:
            path = resource_filename('hlsstack', _MMODEL_SIMILARITY_COG)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    'no bundled similarity raster at %s. Generate one with '
                    'mmodel_sel `scripts/b_embeddings.py --steps export --stack`, '
                    'or pass `similarity=<path>` / `embedding=<64-band raster>`.'
                    % path
                )
        raw = rioxarray.open_rasterio(path)
        if raw.rio.crs is None:
            raise ValueError('similarity raster %r has no CRS' % path)
        raw = raw.where(raw != _MMODEL_SIM_NODATA)
        if np.nanmax(np.abs(raw.values)) > 1.5:      # int16-scaled cosines
            raw = raw / _MMODEL_SIM_SCALE
        raw = raw.rio.reproject_match(ref)
        if raw.sizes['band'] == len(keys):           # no domain band -> synthesize
            dom = xr.full_like(raw.isel(band=0), np.nan)
            raw = xr.concat([raw, dom.expand_dims(band=[len(keys) + 1])], dim='band')
        sim = raw.rename({'band': 'model'})
        sim = sim.assign_coords(model=band_names)

    finite = float(np.isfinite(sim.values).mean())
    if finite < 0.5:
        raise ValueError(
            'the similarity raster covers < 50%% of `dat` after reprojection '
            '(%.0f%% finite) -- check its CRS / extent' % (100 * finite)
        )

    if embedding is None:
        _mmodel_sim_cache[cache_key] = sim
    return sim


def pred_bm_mmodel(dat, model, similarity=None, embedding=None, p=None,
                   domain_mask=True, domain_threshold=None):
    """Drop-in for :func:`pred_bm`: the mmodel_sel embedding-p=2 biomass blend.

    Predicts biomass (kg/ha) with each of the five site-calibrated local PLS
    models distilled into ``model`` and averages the five per pixel with
    ``weights.power_weights(sim_cos, p)`` -- weight ``(cos_i - row-min cos)^p``,
    so the least-similar model contributes nothing. Predictions are conditional
    medians (naive square back-transform), biased low, exactly as ``pred_bm``.

    Parameters
    ----------
    dat : xr.Dataset
        HLS reflectance, band-name variables, dims ``(time, y, x)`` -- the same
        object ``pred_bm`` takes. Intended for
        ``ds.map_blocks(pred_bm_mmodel, template=ds['NIR1'].astype('float32'),
        kwargs=dict(model=bundle))``.
    model : dict
        A loaded bundle, ``models.load.load_model('mmodel_biomass')``.
    similarity : None | str | xr.DataArray
        ``None`` -> the bundled coarse CONUS 6-band cosine raster (package data),
        reprojected to ``dat``'s grid. A path -> a 5- or 6-band cosine raster
        (bands in ``model['model_keys']`` order, optional trailing ``domain``).
        A DataArray with a ``model`` dim -> used as-is (windowed to ``dat``).
    embedding : None | str | xr.DataArray
        A 64-band annual embedding raster; cosines and the domain band are
        computed on the fly from ``model``. Overrides ``similarity``.
    p : float, optional
        Weight exponent; defaults to ``model['weight_p']`` (2.0).
    domain_mask : bool | float
        If truthy, NaN the blend where the domain similarity is below the
        threshold (default ``True``). A float is taken as the threshold
        (implies ``True``). ``False`` disables.
    domain_threshold : float, optional
        Cutoff override; defaults to ``model['domain_similarity_threshold']``
        (0.65).

    Returns
    -------
    xr.DataArray
        ``(time, y, x)`` float32, kg/ha, NaN where unpredictable or out of domain.
    """
    bundle = _mm.load_mmodel_bundle(model)
    keys = list(bundle['model_keys'])
    K = len(keys)
    model_vars = list(bundle['feature_names'])
    missing = set(model_vars) - set(func_dict)
    if missing:
        raise KeyError('pred_bm_mmodel: no index function for %s' % sorted(missing))

    if not isinstance(domain_mask, bool) and isinstance(domain_mask, (int, float)):
        domain_threshold = float(domain_mask)
        domain_mask = True
    dthr = float(bundle['domain_similarity_threshold']
                 if domain_threshold is None else domain_threshold)
    pp = float(bundle['weight_p'] if p is None else p)
    coef = bundle['coef']                       # (K, 28)
    bias = bundle['intercept']                  # (K,)
    inv_lam = 1.0 / float(bundle['lambda_boxcox'])

    sim = _prepare_similarity(dat, bundle, similarity, embedding)

    def pred_func(*args):
        feats, sims = args[:28], args[28:]      # sims: K cosine bands + domain
        grid = feats[0].shape[1:]
        T = feats[0].shape[0]
        npix = int(np.prod(grid))
        max_f32 = np.finfo(np.float32).max

        X_all = np.stack([a.reshape(a.shape[0], -1) for a in feats], axis=-1).astype(np.float32)
        S = np.stack([np.asarray(s).reshape(-1) for s in sims[:K]], axis=-1).astype(np.float64)
        dom = np.asarray(sims[K]).reshape(-1).astype(np.float64)

        w = _mm.power_weights(S, pp)            # (npix, K)
        if domain_mask:
            w[~(np.nan_to_num(dom, nan=-np.inf) >= dthr)] = np.nan

        out = np.full((T, npix), np.nan, dtype=np.float32)
        for t in range(T):
            Xt = X_all[t]
            Xt = np.where(np.isfinite(Xt) & (np.abs(Xt) <= max_f32), Xt, np.nan)
            valid = ~np.isnan(Xt).any(axis=1)
            if valid.any():
                Xv = Xt[valid].astype(np.float64)
                link = Xv @ coef.T + bias
                preds = np.clip(link, 0.0, None) ** inv_lam
                out[t, valid] = _mm.blend(preds, w[valid]).astype(np.float32)
                del Xv, link, preds
            del Xt, valid
        del X_all, S, w
        return out.reshape((T,) + grid)

    feat_das = [func_dict[v](dat) for v in model_vars]
    sim_das = [sim.isel(model=i) for i in range(K + 1)]
    return xr.apply_ufunc(
        pred_func,
        *feat_das, *sim_das,
        dask='parallelized',
        vectorize=False,
        input_core_dims=[['time', 'y', 'x']] * 28 + [['y', 'x']] * (K + 1),
        output_core_dims=[['time', 'y', 'x']],
        output_dtypes=['float32'],
    )


_ABSENT = object()

# Rows used to check that the recovered affine map really reproduces .predict().
_BOOT_PROBE_ROWS = 512

# Pixels per matmul batch. Bounds the (batch, nboot) prediction matrix -- the
# old pd.concat of nboot float64 Series was sized by the whole date instead
# (~3.1 GB for one tbng date, ~5.4 GB for a full HLS tile), which no choice of
# -c/spatial_chunk_size could bound because it is sized before tiling applies.
_BOOT_BATCH = 250_000


def _load_boot_models(mod_boot_dir, nboot):
    """Unpickle a deterministic nboot-member subset of a bootstrap ensemble."""
    paths = sorted(glob.glob(os.path.join(mod_boot_dir, '*.pk')))
    if not paths:
        raise ValueError('no bootstrap models (*.pk) found in ' + str(mod_boot_dir))
    if nboot is not None and nboot < len(paths):
        # A fixed seed, rather than random.sample()'s fresh per-call draw. The
        # old behaviour re-drew a different subset for every block, which made
        # Biomass_SE nondeterministic run-to-run *and* inconsistent between
        # adjacent dates -- the reason create_bm_se cannot trust its own output
        # for change detection and has to borrow Biomass's. Seeding keeps the
        # subset uniformly drawn (the members are exchangeable bootstrap
        # replicates, so any fixed subset is as good as a fresh one) while
        # making it reproducible.
        paths = sorted(random.Random(0).sample(paths, nboot))
    elif nboot is not None and nboot > len(paths):
        print('      pred_bm_se: only %d bootstrap model(s) in %s; using all of '
              'them (nboot=%d requested)' % (len(paths), mod_boot_dir, nboot),
              flush=True)
    models = []
    for p in paths:
        with open(p, 'rb') as f:
            models.append(pd.compat.pickle_compat.load(f))
    return models


def _affine_ensemble(models, feats):
    """
    Collapse a bootstrap ensemble to one (n_features, n_members) matrix.

    Each member is a TransformedTargetRegressor wrapping
    Pipeline([StandardScaler, PLSRegression(scale=False)]) -- scaling and PLS
    are both linear, so the regressor is an exact affine map and all members
    together are a single matmul. Only the target transform (identity on some
    vintages, xfrm_y/bxfrm_y i.e. sqrt/square on others) is nonlinear, and it
    is elementwise, so it applies to the whole (n_pixels, n_members) result
    at once.

    The coefficients are recovered by *probing* -- evaluating the regressor on
    a zero row and the identity basis -- rather than by reading coef_/_x_mean/
    _x_std. These pickles were written by a different sklearn version (they
    raise InconsistentVersionWarning, and their Pipeline repr is already
    broken by an attribute that no longer exists), so their attribute layout
    is not trustworthy; probing reproduces whatever the *installed* sklearn's
    .predict() actually computes, which is the behaviour we must preserve.

    Returns (W, B, inv) or (None, None, None) if the ensemble does not have
    this shape, in which case the caller falls back to per-member .predict().
    """
    n = len(feats)
    invs = set()
    for m in models:
        if list(getattr(m, 'feature_names_in_', [])) != feats:
            return None, None, None
        if not hasattr(m, 'regressor_') or not hasattr(m, 'transformer_'):
            return None, None, None
        invs.add(getattr(m.transformer_, 'inverse_func', _ABSENT))
    if len(invs) != 1:
        return None, None, None
    inv = invs.pop()
    if inv is _ABSENT:
        return None, None, None

    probe = np.zeros((n + 1, n), dtype=np.float64)
    probe[1:] = np.eye(n)
    probe_df = pd.DataFrame(probe, columns=feats)

    W = np.empty((n, len(models)), dtype=np.float64)
    B = np.empty(len(models), dtype=np.float64)
    for k, m in enumerate(models):
        out = np.asarray(m.regressor_.predict(probe_df), dtype=np.float64).ravel()
        if out.size != n + 1:
            return None, None, None
        B[k] = out[0]
        W[:, k] = out[1:] - out[0]

    # Verify the recovered map against the real .predict(), on rows spanning
    # the magnitudes the features actually take. An exactly affine model
    # matches to round-off; anything else fails here and takes the fallback.
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(_BOOT_PROBE_ROWS, n)) * 1000.0
    sample_df = pd.DataFrame(sample, columns=feats)
    recon = sample @ W + B
    if inv is not None:
        recon = inv(recon)
    for k, m in enumerate(models):
        ref = np.asarray(m.predict(sample_df), dtype=np.float64).ravel()
        if not np.allclose(ref, recon[:, k], rtol=1e-6, atol=1e-6):
            return None, None, None
    return W, B, inv


@functools.lru_cache(maxsize=4)
def _boot_ensemble(mod_boot_dir, nboot):
    """
    Load and prepare a bootstrap ensemble once per worker process.

    pred_bm_se used to open and unpickle every member *inside* its kernel --
    once per date, per block, per worker. The CPER ensemble is 100 members of
    2.79 MB, so drawing 50 of them re-read ~140 MB from /project for every
    single date. Almost all of that is x_scores_/y_scores_ (the 8682 training
    rows' latent scores), which inference never touches; the predictive
    content is ~1.2 kB per member.

    Cached per (directory, nboot). On the affine path the member objects are
    dropped once their coefficients are extracted, so what stays resident is
    the ~11 kB (W, B) rather than the ~140 MB of unpickled estimators.
    """
    models = _load_boot_models(mod_boot_dir, nboot)
    feats = list(models[0].feature_names_in_)
    W, B, inv = _affine_ensemble(models, feats)
    if W is None:
        print('      pred_bm_se: bootstrap ensemble in %s is not an affine PLS '
              'ensemble -- falling back to per-member predict()'
              % mod_boot_dir, flush=True)
        return {'feats': feats, 'W': None, 'B': None, 'inv': None,
                'models': models, 'n': len(models)}
    return {'feats': feats, 'W': W, 'B': B, 'inv': inv,
            'models': None, 'n': len(models)}


def pred_bm_se(dat, model, mod_boot_dir, nboot=100, avg_std=144.61,
               boot_batch=_BOOT_BATCH):
    """
    Per-pixel standard error of the biomass prediction, from a bootstrapped
    PLS ensemble. See https://doi.org/10.1016/j.jbusres.2016.03.049

    Measured 27-30x faster than the pre-2026-08 implementation (27.4s -> 1.0s
    for one 1 Mpx date; 112.0s -> 3.7s at 4 Mpx), which stacked y/x into a
    pandas MultiIndex before computing the 28 index features, re-unpickled 50
    ensemble members from disk inside the kernel on every call, repeated a
    loop-invariant dropna once per member, and accumulated the result in an
    n_pixels x nboot float64 pd.concat. Full write-up, including the
    equivalence testing, in the py_hls_nrt repo:
    docs/bug_fixes/predict_stack_multiindex.md, "Follow-up: pred_bm_se".

    The ensemble subset is seeded, so repeated runs on identical input agree;
    it used to be redrawn at random per block. See _load_boot_models.
    """
    ens = _boot_ensemble(mod_boot_dir, nboot)
    model_vars = list(model.feature_names_in_)

    # The member features have to come from the arrays we compute for the
    # point model. The old implementation required the same thing implicitly:
    # it assigned the member-level dropna result into the point model's
    # all-finite mask, which only lines up when the two feature sets agree.
    missing = [v for v in ens['feats'] if v not in model_vars]
    if missing:
        raise ValueError(
            'bootstrap ensemble in {} needs feature(s) {} that the point model '
            'does not provide'.format(mod_boot_dir, missing))
    col_idx = [model_vars.index(v) for v in ens['feats']]
    W, B, inv = ens['W'], ens['B'], ens['inv']
    boot_models = ens['models']

    def _se_batch(X):
        """Per-pixel SE across the ensemble for one (n_pixels, n_feat) batch."""
        if W is not None:
            preds = X @ W + B
            if inv is not None:
                preds = inv(preds)
        else:
            X_df = pd.DataFrame(X, columns=ens['feats'])
            preds = np.empty((X.shape[0], len(boot_models)), dtype=np.float64)
            for k, mod_tmp in enumerate(boot_models):
                preds[:, k] = np.asarray(mod_tmp.predict(X_df)).ravel()
        # ddof=1 to match the pandas DataFrame.std() this replaces; numpy
        # defaults to ddof=0 and would quietly shrink every SE.
        return preds.std(axis=1, ddof=1) + avg_std

    def pred_func(*args):
        # args each arrive as (time, y, x). Flatten y/x with numpy, one date at
        # a time, rather than upstream with xarray's .stack(z=('y','x')) -- see
        # pred_func_xr below for why.
        grid = args[0].shape[1:]
        time_steps = args[0].shape[0]
        n_pixels = int(np.prod(grid))
        out = np.full((time_steps, n_pixels), np.nan, dtype=np.float32)

        for t in range(time_steps):
            flat = [np.asarray(a[t]).reshape(-1) for a in args]
            # A pixel is usable where every feature is finite. Accumulated one
            # feature at a time so nothing of (n_features, n_pixels) size is
            # ever materialized at once.
            valid = np.ones(n_pixels, dtype=bool)
            for f in flat:
                valid &= np.isfinite(f)
            idx = np.flatnonzero(valid)
            if idx.size == 0:
                continue

            for start in range(0, idx.size, boot_batch):
                sl = idx[start:start + boot_batch]
                X = np.empty((sl.size, len(col_idx)), dtype=np.float64)
                for c, j in enumerate(col_idx):
                    X[:, c] = flat[j][sl]
                out[t, sl] = _se_batch(X).astype(np.float32)
                del X

            del flat, valid, idx

        return out.reshape((time_steps,) + grid)

    def pred_func_xr(dat_xr, model_vars_xr):
        # The index functions run on the native (time, y, x) grid. They used to
        # run on dat_xr.stack(z=('y','x')), which builds a pandas MultiIndex
        # with one entry per pixel -- and every elementwise op inside func_dict
        # then pays index alignment proportional to it. Same fix, and the same
        # measurement, as pred_bm and pred_cov: on one tbng-sized date
        # (3000x2600, 7.8 Mpx) 14 index functions took 25.3s stacked versus
        # 0.76s unstacked. This model uses 28 of them, so it was paying roughly
        # double. pred_func flattens y/x with numpy instead, where it is a view.
        vars_list_xr = [func_dict[v](dat_xr) for v in model_vars_xr]

        return xr.apply_ufunc(
            pred_func,
            *vars_list_xr,
            dask='parallelized',
            vectorize=False,
            input_core_dims=[['time', 'y', 'x']] * len(model_vars_xr),
            output_core_dims=[['time', 'y', 'x']],
            output_dtypes=['float32']
        )

    return pred_func_xr(dat, model_vars)


def xr_cdf(dat):
    return xr.apply_ufunc(st.norm.cdf, dat)


def pred_bm_thresh(dat_bm, dat_se, thresh_kg):
    #thresh_log = np.log(thresh_kg)
    dat_bm = dat_bm.stack(z=('y', 'x'))
    dat_se = dat_se.stack(z=('y', 'x'))

    def pred_func(arr_bm, arr_se):
        #thresh_pre = (thresh_log - np.log(arr_bm)) / arr_se
        thresh_pre = (thresh_kg - arr_bm) / arr_se
        arr_thresh = st.norm.cdf(thresh_pre)
        return arr_thresh.astype('float32')

    def pred_func_xr(dat_bm, dat_se):

        thresh_xr = xr.apply_ufunc(pred_func,
                               *[dat_bm, dat_se],
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=[['z'], ['z']],
                               output_core_dims=['z'],
                               output_dtypes=['float32'])
        return thresh_xr.unstack('z')

    # bm_out = pred_func_xr(dat_masked, model_vars, dims_list)
    thresh_out = pred_func_xr(dat_bm, dat_se)
    return thresh_out


def pred_cov(dat, model):
    model_vars = model.feature_names_in_
    band_list = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2',
                 'DFI', 'NDVI', 'NDTI', 'SATVI', 'NDII7',
                 'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346']

    def pred_cov_np(*args):
        # args are each (time, y, x) — flatten y/x to one pixel axis with numpy
        # (see pred_cov_xr below for why not xarray's .stack) and stack along a
        # new last axis to get (time, pixel, bands)
        grid = args[0].shape[1:]
        mat = np.stack([a.reshape(a.shape[0], -1) for a in args], axis=-1).astype(np.float32)

        time_steps, n_pixels, n_bands = mat.shape
        max_f32 = np.finfo(np.float32).max

        # Output: (4, time, pixel)
        unmixed = np.full((4, time_steps, n_pixels), np.nan, dtype=np.float32)
        
        for t in range(time_steps):
            mat_t = mat[t]  # (z, bands)
            mat_t = np.where(np.isfinite(mat_t) & (np.abs(mat_t) <= max_f32), mat_t, np.nan)
            #valid_mask = ~np.any(np.isnan(mat_t), axis=1)

            # Restore DataFrame to satisfy StandardScaler's feature name expectation
            df_t = pd.DataFrame(mat_t, columns=model_vars)
            valid_mask = ~df_t.isna().any(axis=1).values
            
            if valid_mask.any():
                #preds = model.predict(mat_t[valid_mask, :]).astype(np.float32)
                preds = model.predict(df_t[valid_mask]).astype(np.float32)
                np.clip(preds, 0, 1, out=preds)
                unmixed[:, t, valid_mask] = preds.T
                del preds
            
            del mat_t, valid_mask
        
        del mat
        unmixed = unmixed.reshape((4, time_steps) + grid)
        return unmixed[0], unmixed[1], unmixed[2], unmixed[3]  # each (time, y, x)

    def pred_cov_xr(dat_xr, name):
        # No dat_xr.stack(z=('y','x')) here -- the index functions run on the
        # native (time, y, x) grid, and pred_cov_np flattens y/x with numpy
        # instead. Stacking builds a pandas MultiIndex with one entry per pixel,
        # and every elementwise op in the 17 func_dict calls below then pays
        # index alignment against it: 46.7s versus 4.9s per tbng-sized date
        # (3000x2600). Bit-identical either way. Same fix as pred_bm -- see its
        # pred_func_xr for the full measurement.
        vars_list_xr = [func_dict[v](dat_xr) for v in band_list]

        # Use separate unique dimension names for each output
        unmixed_xr = xr.apply_ufunc(
            pred_cov_np,
            *vars_list_xr,
            dask='parallelized',
            vectorize=False,
            input_core_dims=[['time', 'y', 'x']] * len(band_list),
            output_core_dims=[['time', 'y', 'x']] * 4,
            output_dtypes=['float32'] * 4
        )

        cov_xr = xr.concat(unmixed_xr, dim='type')
        cov_xr = cov_xr.assign_coords(type=name)
        return cov_xr.to_dataset(dim='type')

    return pred_cov_xr(dat, name=['BARE', 'SD', 'GREEN', 'LITT'])


def pred_cp(dat, model):
    import ctypes
    time_chunks = dat.chunks[dat.dims.index('time')]
    if len(time_chunks) != 1:
        raise ValueError(
            f"Data must not be chunked along 'time'. "
            f"Found {len(time_chunks)} time chunks. "
            f"Re-chunk with dat.chunk({{'time': -1}}) before calling pred_cp()."
        )
        
    feature_cols = ['NDVI', 'NDVI_d30', 'iNDVI', 't_SOS', 'iNDVI_dry']
    
    def running_mean(x, N):
        cumsum = np.nancumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def pheno_fq_metrics_vectorized(ndvi_ts_mean):
        b = len(ndvi_ts_mean)

        if (np.sum(np.isnan(ndvi_ts_mean)) >= b * 0.5) or (np.sum(~np.isnan(ndvi_ts_mean[10:75])) == 0):
            return None

        try:
            ndvi_thresh1 = np.nanpercentile(ndvi_ts_mean[91:201], 40.0)
            date_thresh1 = next(x for x in np.where(ndvi_ts_mean > ndvi_thresh1)[0] if x > 30)

            dndvi_ts_mean = np.ones(b) * np.nan
            dndvi_ts_mean[25:] = running_mean(np.diff(ndvi_ts_mean), 25)

            mask = dndvi_ts_mean[:date_thresh1] > 0
            dndvi_thresh2 = np.nanpercentile(dndvi_ts_mean[:date_thresh1][mask], 35.0)
            sos = np.where(dndvi_ts_mean[:date_thresh1] < dndvi_thresh2)[0][-1]
            ndvi_base = np.nanmean(ndvi_ts_mean[10:75])

            # IGR: vectorized cumulative sum over rolling 30-day window
            ndvi_d1 = np.diff(ndvi_ts_mean, prepend=ndvi_ts_mean[0])
            # Use stride tricks for rolling sum instead of loop
            d1_cumsum = np.nancumsum(ndvi_d1)
            d1_cumsum_padded = np.concatenate([[0] * 30, d1_cumsum])
            ndvi_d1_cum30 = d1_cumsum_padded[30:] - d1_cumsum_padded[:b]
            ndvi_d1_cum30[np.isnan(ndvi_d1)] = np.nan

            # Vectorized integrated NDVI (replaces Python loop)
            ts_tmp = ndvi_ts_mean - ndvi_base
            ts_tmp_cumsum = np.nancumsum(ts_tmp)
            ndvi_int_ts = np.zeros(b)
            ndvi_int_ts[sos:] = ts_tmp_cumsum[sos:] - ts_tmp_cumsum[sos] + ts_tmp[sos]
            ndvi_int_ts[:sos] = 0.0

            # Rate of change
            ndvi_rate_ts = np.zeros(b)
            denom = np.arange(b) - sos + 1
            ndvi_rate_ts[sos:] = ndvi_int_ts[sos:] / denom[sos:]

            # Vectorized dry biomass (replaces Python loop)
            ndvi_max_running = np.maximum.accumulate(
                np.where(np.isnan(ndvi_ts_mean), -np.inf, ndvi_ts_mean)
            )
            ndvi_max_running[:sos] = np.nan

            decline = ndvi_d1 < 0
            ndvi_dry_ts = np.zeros(b)
            valid = decline & (ndvi_max_running > 0) & (np.arange(b) >= sos)
            ndvi_dry_ts[valid] = (
                -ndvi_d1[valid] / ndvi_max_running[valid]
            ) * ndvi_int_ts[valid]

            ndvi_int_dry_ts = np.nancumsum(ndvi_dry_ts)
            ndvi_int_dry_ts[:sos] = 0.0

            ndvi_int_dry_pct_ts = np.zeros(b)
            nonzero = ndvi_int_ts != 0
            ndvi_int_dry_pct_ts[nonzero] = ndvi_int_dry_ts[nonzero] / ndvi_int_ts[nonzero]

            t_sos = np.arange(b) - sos

            return {
                'NDVI': ndvi_ts_mean,
                'NDVI_d30': ndvi_d1_cum30,
                'iNDVI': ndvi_int_ts,
                'iNDVI_dry': ndvi_int_dry_ts,
                'NDVI_rate': ndvi_rate_ts,
                'iNDVI_dry_pct': ndvi_int_dry_pct_ts,
                'SOS_doy': sos,
                't_SOS': t_sos,
            }

        except Exception:
            return None

    def pred_func_block(block):
        time_ax = block.dims.index('time')
        arr = np.moveaxis(block.values, time_ax, -1)
        x, y, t = arr.shape
        pixels = arr.reshape(-1, t)
        out = np.full_like(pixels, np.nan, dtype='float32')
    
        # compute pheno metrics for all pixels first
        pheno_list = [pheno_fq_metrics_vectorized(px) if not np.all(np.isnan(px)) else None 
                      for px in pixels]
    
        # collect valid indices and build batch feature matrix
        valid_idx = [i for i, p in enumerate(pheno_list) if p is not None]
        
        if valid_idx:
            # check for NaNs per pixel and further filter
            valid_idx = [i for i in valid_idx 
                         if not np.any(np.isnan(np.column_stack([
                             pheno_list[i]['NDVI'], pheno_list[i]['NDVI_d30'],
                             pheno_list[i]['iNDVI'], pheno_list[i]['t_SOS'],
                             pheno_list[i]['iNDVI_dry']
                         ])))]
    
        if valid_idx:
            batch_features = pd.DataFrame(
                np.vstack([
                    np.column_stack([
                        pheno_list[i]['NDVI'], pheno_list[i]['NDVI_d30'],
                        pheno_list[i]['iNDVI'], pheno_list[i]['t_SOS'],
                        pheno_list[i]['iNDVI_dry'],
                    ])
                    for i in valid_idx
                ]),
                columns=feature_cols
            )
    
            try:
                # one predict call for entire chunk
                all_preds = model.predict(batch_features).astype('float32')
                
                # split back into per-pixel arrays of length t
                # each pixel contributed exactly t rows so this is clean
                split_preds = np.split(all_preds, len(valid_idx))
                
                for j, i in enumerate(valid_idx):
                    cp_pred = split_preds[j]  # shape (t,)
                    cp_smooth = np.convolve(cp_pred, np.ones(7) / 7.0, mode='full')[:t]
                    cp_smooth[:6] = np.nan
                    cp_smooth[pheno_list[i]['t_SOS'] < 0] = np.nan
                    out[i] = cp_smooth
            except Exception as e:
                print(e)
    
        result = np.moveaxis(out.reshape(x, y, t), -1, time_ax)
        return block.copy(data=result)
    
    
    def pred_func_xr(dat_xr):
        return dat_xr.map_blocks(pred_func_block, template=dat_xr)

    return pred_func_xr(dat)


def pred_cp_old(dat, model_name):
    if type(model_name) != str:
        print('ERROR: pass only the model name as a string, not the entire model object.')
    #dat_masked = dat.where(dat.notnull())

    def pheno_fq_metrics(ndvi_ts_mean):
    
    
        """
        ndvi_ts_mean (1-d array): time series of NDVI values for an entire calendar year (e.g., mean for a single pasture)
        produce_ts (boolean): whether to return the entire time series (default) or just average between b_start and b_end (see below)
        b_start (int): the day of year for the start of the time series subset to average over for output. Only used if produce_ts==False.
        b_end (int): the day of the year for the end of the time series subset to average over for output. Only used if produce_ts==False.
        """
    
        def running_mean(x, N):
            cumsum = np.nancumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)
    
        def ndvi_int_calc(ts, base, sos):
            ts_tmp = ts - base
            ndvi_int_ts = np.ones_like(ts_tmp) * np.nan
            for b_i in range(ts_tmp.shape[0]):
                ndvi_int_ts[b_i] = np.nansum(ts_tmp[sos:b_i + 1])
            return ndvi_int_ts

        # get length of time series
        b = len(ndvi_ts_mean)
        
        if (sum(np.isnan(ndvi_ts_mean)) < b*0.5) and (sum(~np.isnan(ndvi_ts_mean[10:75])) > 0):
            try:
                # calculate start of season and base ndvi
                ndvi_thresh1 = np.nanpercentile(ndvi_ts_mean[91:201], 40.0)
                date_thresh1 = next(x for x in np.where(ndvi_ts_mean > ndvi_thresh1)[0] if x > 30)
                dndvi_ts_mean = np.ones_like(ndvi_ts_mean) * np.nan
                dndvi_ts_mean[25:] = running_mean(np.diff(ndvi_ts_mean), 25)
                dndvi_thresh2 = np.nanpercentile(dndvi_ts_mean[:date_thresh1][dndvi_ts_mean[:date_thresh1] > 0], 35.0)
                sos = np.where(dndvi_ts_mean[:date_thresh1] < dndvi_thresh2)[0][-1]
                ndvi_base = np.nanmean(ndvi_ts_mean[10:75])
            
                # calculate 'instantaneous greenup rate (IGR)' with potentially different lags
                ndvi_ts_smooth_d1 = np.diff(ndvi_ts_mean, prepend=ndvi_ts_mean[0])
            
                ndvi_ts_smooth_d1_cum30 = np.empty_like(ndvi_ts_smooth_d1)
                for i in range(b):
                    ndvi_ts_smooth_d1_cum30[i] = np.nansum(ndvi_ts_smooth_d1[i - 30:i], axis=0)
            
                # cleanup and reshape IGR metrics
                ndvi_ts_smooth_d1_cum30[np.where(np.isnan(ndvi_ts_smooth_d1))] = np.nan
            
                # calculate integrated ndvi
                ndvi_int_ts = ndvi_int_calc(ndvi_ts_mean, ndvi_base, sos)
                ndvi_rate_ts = np.zeros_like(ndvi_int_ts)
            
                # calculate rate of change
                ndvi_rate_ts[sos:] = ndvi_int_ts[sos:] / (range(sos, ndvi_int_ts.shape[0]) - sos + 1)
            
                # calculate percent dry biomass estimate
                ndvi_dry_ts = np.zeros_like(ndvi_int_ts)
                ndvi_int_dry_ts = np.zeros_like(ndvi_int_ts)
                for i in range(sos, ndvi_dry_ts.shape[0]):
                    if ndvi_ts_smooth_d1[i] < 0:
                        ndvi_dry_ts[i] = (-1.0 * ndvi_ts_smooth_d1[i] / np.nanmax(ndvi_ts_mean[:i])) * ndvi_int_ts[i]
                    ndvi_int_dry_ts[i] = np.nansum(ndvi_dry_ts[:i])
            
                ndvi_int_dry_pct_ts = np.zeros_like(ndvi_int_ts)
                ndvi_int_dry_pct_ts[ndvi_int_ts != 0] = ndvi_int_dry_ts[ndvi_int_ts != 0] / ndvi_int_ts[ndvi_int_ts != 0]
                
                # create the output dataframe
                df_out = pd.DataFrame(
                    {
                        'NDVI': ndvi_ts_mean,
                        'NDVI_d30': ndvi_ts_smooth_d1_cum30,
                        'iNDVI':ndvi_int_ts,
                        'iNDVI_dry': ndvi_int_dry_ts,
                        'NDVI_rate': ndvi_rate_ts,
                        'iNDVI_dry_pct': ndvi_int_dry_pct_ts,
                        'SOS_doy': sos,
                        't_SOS': np.arange(b) - sos
                    }
                )
                return df_out
            # return all NaN values if an IndexError occurs - this is usually due to inability to get thresholds
            #except IndexError:
            except Exception as e: 
                #print(e)
                return np.ones_like(ndvi_ts_mean) * np.nan
        else:
            return np.ones_like(ndvi_ts_mean) * np.nan
    
    def pred_func(ndvi_ts):
        if np.all(np.isnan(ndvi_ts)):
            #print('all NaN found')
            cp_out = np.ones_like(ndvi_ts) * np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                from hlsstack.models.load import load_model
                model = load_model(model_name)  # each worker loads its own copy
                # create the phenologic metrics
                df_pheno = pheno_fq_metrics(ndvi_ts)
                # apply the model (if successfully output phenologic metrics)
                if type(df_pheno) is pd.DataFrame and not np.any(np.isnan(df_pheno)):
                    # get the features for the model
                    cp_features = df_pheno[['NDVI', 'NDVI_d30', 'iNDVI', 't_SOS', 'iNDVI_dry']]
                    try:
                        # apply the random forest model
                        df_pheno['CP_pred'] = model.predict(cp_features)
                        df_pheno['CP_pred'] = df_pheno['CP_pred'].rolling(7, center=False).mean()
                        cp_out = df_pheno['CP_pred'].values
                        cp_out[df_pheno['t_SOS'] < 0] = np.nan
                    except Exception as e: 
                        print(e)
                        print("An error occurred!")
                        cp_out = np.ones_like(ndvi_ts) * np.nan
                
                # if outputting the phenologic metrics failed, return all NaN values
                else:
                    cp_out = np.ones_like(ndvi_ts) * np.nan
        return cp_out

    def pred_func_xr(dat_xr):
        cp_xr = xr.apply_ufunc(pred_func,
                               dat_xr,
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=[['time']],
                               output_core_dims=[['time']],
                               output_dtypes=['float32'],
                              )
        return cp_xr

    cp_out_xr = pred_func_xr(dat)

    return cp_out_xr