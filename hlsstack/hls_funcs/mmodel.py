"""Standalone pieces of the mmodel_sel embedding-p=2 biomass blend.

`pred_bm_mmodel` (in ``predict.py``) is a drop-in for `pred_bm` that runs the
five site-calibrated local PLS models from the ``mmodel_sel`` project and blends
their per-pixel predictions by similarity in Google Satellite Embedding space.
This module holds the parts that let `hlsstack` do that **without importing
mmodel_sel** at runtime:

* ``power_weights`` / ``blend`` -- the model-averaging kernel, vendored verbatim
  from ``mmodel_sel/src/mmodel_sel/weights.py`` @ 5358de02ba90 (2026-08-31).
  ``w_i ∝ (cos_i - row-min cos)^p`` renormalized, so the least-similar model gets
  exactly zero weight. If mmodel_sel's kernel changes, re-vendor here and
  regenerate the bundle (mmodel_sel stage I).
* ``blend_predictive_sd`` -- the per-pixel standard error of prediction for the
  blend, vendored from ``mmodel_sel/src/mmodel_sel/sep.py`` @ ed25177bde20
  (2026-08-31). Consumed by ``pred_bm_mmodel_se``; needs a ``sep`` block in the
  bundle (``format_version`` 2).
* ``load_mmodel_bundle`` -- read + validate the distilled bundle
  (``models/mmodel_biomass_bundle_*.pk``): a plain dict of numpy arrays, no
  mmodel_sel or sklearn classes. Registered in ``models/load.py`` as
  ``'mmodel_biomass'``. ``format_version`` 1 is a point-estimate bundle; 2 adds
  the ``sep`` block.
* ``cosine_from_embedding`` -- cosine similarity of a 64-band annual embedding
  raster to each model's training-footprint mean vector, plus the
  nearest-training-neighbour domain similarity. Only needed for the
  ``embedding=`` path; the default path reads a pre-reduced similarity raster.

numpy only -- this stays a leaf module (see CLAUDE.md module map), because
``predict.py`` imports it alongside ``bands``/``indices`` for ``pred_bm_mmodel``.
"""
import numpy as np
import joblib

BUNDLE_FORMAT = "mmodel_sel.biomass_blend"


# --------------------------------------------------------------------- weights
# Vendored from mmodel_sel/src/mmodel_sel/weights.py @ 5358de02ba90.


def _renormalize(w):
    total = np.nansum(w, axis=1, keepdims=True)
    return np.where(total > 0, w / np.where(total == 0, np.nan, total), np.nan)


def power_weights(scores, p=2.0, floor=None):
    """``w ∝ (s - floor)^p`` -- the similarity analogue of inverse-distance weighting.

    ``floor`` defaults to the per-row minimum score, so the least-similar model
    always gets zero weight. Rows where every model ties fall back to equal
    weighting; rows where no model is finite come back all-NaN.
    """
    s = np.where(np.isfinite(scores), scores, np.nan)
    base = np.nanmin(s, axis=1, keepdims=True) if floor is None else floor
    shifted = np.nan_to_num(np.clip(s - base, 0.0, None), nan=0.0) ** p

    degenerate = shifted.sum(axis=1) == 0
    if degenerate.any():
        shifted = shifted.copy()
        shifted[degenerate] = np.isfinite(scores[degenerate]).astype(float)
    return _renormalize(shifted)


def blend(preds, w):
    """Weighted average of per-model predictions, ignoring models that returned NaN."""
    usable = np.isfinite(preds) & np.isfinite(w)
    w_eff = np.where(usable, np.nan_to_num(w, nan=0.0), 0.0)
    total = w_eff.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(usable, preds * w_eff, 0.0).sum(axis=1) / total
    return np.where(total > 0, out, np.nan)


# --------------------------------------------------------------- standard error
# Vendored from mmodel_sel/src/mmodel_sel/sep.py @ ed25177bde20.


def blend_predictive_sd(g, w, sigma, p, q, b, k2, f0):
    """Per-row standard error of prediction (kg/ha) for the p=2 blend.

    ``SEP = sqrt(p*A + q*B + b*V_select + k2*ybar**2 + f0**2)`` -- within-model
    residual propagated through the square back-transform by the delta method
    (``d ybar / d eta_i = 2 w_i g_i``), plus between-model disagreement, plus a
    calibrated transfer floor.

    Parameters
    ----------
    g : ndarray ``(n, K)``
        Non-negative link-scale member predictions, i.e. ``clip(X @ coef.T +
        intercept, 0)`` (equivalently ``sqrt`` of the kg/ha members).
    w : ndarray ``(n, K)``
        Blend weights (rows sum to 1; NaN where a model is unusable).
    sigma : ndarray ``(K,)``
        Per-model out-of-fold link-scale residual SD (bundle ``sep['sigma_link']``).
    p, q, b, k2, f0 : float
        Calibrated coefficients (bundle ``sep`` block).

    It is a residual + model-disagreement error with a calibrated transfer
    floor, not a full parameter-uncertainty interval, and it is a spread about
    the blend's biased-low conditional median (the point prediction), not the
    mean.
    """
    g = np.asarray(g, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    wv = np.where(np.isfinite(w), w, 0.0)

    yhat = g ** 2
    ybar = np.sum(wv * yhat, axis=1)
    js = (2.0 * wv * g) * sigma                       # J_i * sigma_i
    A = np.sum(js ** 2, axis=1)
    B = np.sum(js, axis=1) ** 2
    v_select = np.sum(wv * (yhat - ybar[:, None]) ** 2, axis=1)

    var = p * A + q * B + b * v_select + k2 * ybar ** 2 + float(f0) ** 2
    return np.sqrt(np.clip(var, 0.0, None))


# ---------------------------------------------------------------------- bundle


def load_mmodel_bundle(path):
    """Load + validate a distilled mmodel_sel biomass-blend bundle.

    Accepts a path, or a dict that is already a loaded bundle (idempotent, so
    callers can pass either the registry name's `load_model(...)` result or a
    path).
    """
    b = path if isinstance(path, dict) else joblib.load(path)
    if b.get("format") != BUNDLE_FORMAT:
        raise ValueError(
            f"{path!r} is not an mmodel_sel biomass bundle "
            f"(format={b.get('format')!r}); expected {BUNDLE_FORMAT!r}. "
            "Regenerate it with mmodel_sel scripts/i_export_mmodel_bundle.py."
        )
    k = len(b["model_keys"])
    coef = np.asarray(b["coef"], dtype=np.float64)
    if coef.shape != (k, 28):
        raise ValueError(f"bundle coef shape {coef.shape}, expected ({k}, 28)")
    if len(b["feature_names"]) != 28:
        raise ValueError(f"bundle has {len(b['feature_names'])} feature names, expected 28")
    mu_mean = np.asarray(b["mu_mean"], dtype=np.float64)
    if not np.allclose(np.linalg.norm(mu_mean, axis=1), 1.0, atol=1e-4):
        raise ValueError("bundle mu_mean rows are not unit-norm")
    b["coef"] = coef
    b["intercept"] = np.asarray(b["intercept"], dtype=np.float64)
    b["mu_mean"] = mu_mean
    b["mu_years"] = np.asarray(b["mu_years"], dtype=np.float64)
    b["mu_year_mask"] = np.asarray(b["mu_year_mask"], dtype=bool)
    b["train_embedding"] = np.asarray(b["train_embedding"], dtype=np.float64)
    if "link_resid_sd" in b:
        b["link_resid_sd"] = np.asarray(b["link_resid_sd"], dtype=np.float64)
    if "n_train" in b:
        b["n_train"] = np.asarray(b["n_train"], dtype=np.int64)

    # format_version 2: the standard-error-of-prediction block (mmodel_sel
    # scripts/i_sep_calibrate.py). Consumed by pred_bm_mmodel_se.
    if "sep" in b:
        s = dict(b["sep"])
        sig = np.asarray(s["sigma_link"], dtype=np.float64)
        if sig.shape != (k,):
            raise ValueError(f"bundle sep sigma_link shape {sig.shape}, expected ({k},)")
        s["sigma_link"] = sig
        for key in ("p", "q", "b", "k2", "f0"):
            s[key] = float(s[key])
        b["sep"] = s
    return b


def _l2_normalize_last(arr):
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm = np.where(norm == 0, np.nan, norm)
    return arr / norm


def cosine_from_embedding(emb, bundle, year=None, domain_block=128):
    """Cosine similarity of an embedding raster to each model + the domain band.

    Parameters
    ----------
    emb : ndarray
        ``(64, y, x)`` -- a single (typically multi-year composite) annual
        embedding image -- or ``(n_year, 64, y, x)`` with ``year`` giving the
        corresponding calendar years for year-matched cosines.
    bundle : dict
        A loaded bundle (``load_mmodel_bundle``).
    year : sequence of int, optional
        Years for the leading axis of a 4-D ``emb``. Ignored for 3-D ``emb``.
    domain_block : int
        Training vectors per matmul when reducing the nearest-neighbour max.

    Returns
    -------
    ndarray
        ``(K + 1, y, x)`` -- one cosine band per model (bundle order) followed by
        the nearest-training-neighbour domain similarity.
    """
    emb = np.asarray(emb, dtype=np.float64)
    keys = list(bundle["model_keys"])
    mu_mean = bundle["mu_mean"]                       # (K, 64)

    if emb.ndim == 4:
        years = list(year) if year is not None else list(bundle["mu_year_list"])
        _, _, ny, nx = emb.shape
        # per-pixel unit vectors, per year
        unit = _l2_normalize_last(np.moveaxis(emb, 1, -1))          # (n_year, y, x, 64)
        sims = np.full((len(keys), ny, nx), np.nan)
        mu_years = bundle["mu_years"]
        mu_mask = bundle["mu_year_mask"]
        bundle_years = list(bundle["mu_year_list"])
        for ki in range(len(keys)):
            acc = np.zeros((ny, nx))
            cnt = np.zeros((ny, nx))
            for yi, yr in enumerate(years):
                if yr not in bundle_years:
                    continue
                bj = bundle_years.index(yr)
                if not mu_mask[ki, bj]:
                    continue
                s = unit[yi] @ mu_years[ki, bj]
                ok = np.isfinite(s)
                acc[ok] += s[ok]
                cnt[ok] += 1
            with np.errstate(invalid="ignore"):
                sims[ki] = np.where(cnt > 0, acc / cnt, np.nan)
        composite = _l2_normalize_last(np.nanmean(np.moveaxis(emb, 1, -1), axis=0))
    else:
        vec = np.moveaxis(emb, 0, -1)                                # (y, x, 64)
        composite = _l2_normalize_last(vec)
        sims = np.stack([composite @ mu_mean[ki] for ki in range(len(keys))], axis=0)

    # nearest-training-neighbour domain similarity, reduced in blocks
    train = bundle["train_embedding"]                                # (N, 64)
    ny, nx, _ = composite.shape
    dom = np.full((ny, nx), -np.inf)
    for start in range(0, len(train), domain_block):
        block = train[start:start + domain_block]                    # (b, 64)
        d = composite @ block.T                                      # (y, x, b)
        dom = np.fmax(dom, np.nanmax(d, axis=-1))
    dom[~np.isfinite(dom)] = np.nan

    return np.concatenate([sims, dom[None]], axis=0)
