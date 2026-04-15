import numpy as np
from scipy.signal import savgol_filter
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
import dask
import dask.array as da_dask
from dask.diagnostics import ProgressBar
import multiprocessing


def despike_ts_xr(da, dat_thresh, days_thresh, z_thresh=3.5, mask_outliers=False, iters=2):
    """
    Vectorized despiking for Xarray/Dask.
    Processes the entire (Time, Y, X) cube at once without Python loops.
    """
    ds_cln = da.copy()

    # 1. Vectorized Modified Z-Score (Outlier Masking)
    if mask_outliers:
        # Calculate median and MAD across the 'time' dimension for every pixel
        median_int = ds_cln.median(dim='time')
        mad_int = np.abs(ds_cln - median_int).median(dim='time')
        
        # Avoid division by zero if MAD is 0
        mad_int = mad_int.where(mad_int != 0, np.nan)
        mod_z = 0.6745 * (ds_cln - median_int) / mad_int
        
        # Mask where robust Z-score > threshold
        ds_cln = ds_cln.where(np.abs(mod_z) <= z_thresh)

    # 2. Setup Time Indices for dx calculations
    # Create a 1D array of indices [0, 1, 2...] and broadcast to 3D
    t_idx = xr.DataArray(np.arange(len(da.time)), dims='time', coords={'time': da.time})
    current_idx = t_idx.broadcast_like(ds_cln)

    # 3. Iterative Despiking (Vectorized)
    for _ in range(iters):
        # Find 'idx_pre' and 'idx_post' using ffill/bfill
        # We mask the grid with current valid values so fill finds the "nearest clear"
        valid_idx_grid = current_idx.where(ds_cln.notnull())
        
        # Shift is required so we don't pick the current pixel as its own neighbor
        idx_pre = valid_idx_grid.ffill(dim='time').shift(time=1)
        idx_post = valid_idx_grid.bfill(dim='time').shift(time=-1)
        
        val_pre = ds_cln.ffill(dim='time').shift(time=1)
        val_post = ds_cln.bfill(dim='time').shift(time=-1)

        # Vectorized Math: dy, dx, and interp
        dx = idx_post - idx_pre
        dy = val_post - val_pre
        slope = dy / dx.where(dx != 0)
        
        dat_interp = val_pre + slope * (current_idx - idx_pre)
        dat_diff = dat_interp - ds_cln
        
        # Calculate shadow_val (handle dy=0 case)
        shadow_val = dat_diff / dy.where(dy != 0)

        # Define the spike mask based on your specific conditions
        is_spike = (
            (dx < days_thresh) & 
            (np.abs(dat_diff) > dat_thresh) & 
            (np.abs(shadow_val) > 2)
        )
        
        # Apply the mask: set spikes to NaN for the next iteration
        ds_cln = ds_cln.where(~is_spike)

    return ds_cln


# ── 1. Vectorized non-uniform Savitzky-Golay (NumPy, no Python loops) ─────────

def non_uniform_savgol(x, y, window, polynom):
    """
    Savitzky-Golay filter for non-uniformly spaced data.
    Vectorized with NumPy — no inner Python loops.
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')
    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')
    if not isinstance(window, (int, np.integer)):
        raise TypeError('"window" must be an integer')
    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')
    if not isinstance(polynom, (int, np.integer)):
        raise TypeError('"polynom" must be an integer')
    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    poly_order  = polynom + 1          # number of coefficients
    n           = len(x)
    y_smoothed  = np.full(n, np.nan)

    # Build ALL local-x windows at once: shape (n_interior, window)
    idx      = np.arange(half_window, n - half_window)          # centre indices
    offsets  = np.arange(window) - half_window                  # -hw … +hw
    win_idx  = idx[:, None] + offsets                           # (n_interior, window)
    t        = x[win_idx] - x[idx, None]                       # local x, centred at 0

    # Vandermonde design matrices: shape (n_interior, window, poly_order)
    powers   = np.arange(poly_order)
    A        = t[:, :, None] ** powers[None, None, :]           # (n_i, w, p)
    tA       = A.transpose(0, 2, 1)                             # (n_i, p, w)

    tAA      = tA @ A                                           # (n_i, p, p)
    tAA_inv  = np.linalg.inv(tAA)                               # (n_i, p, p)
    coeffs   = tAA_inv @ tA                                     # (n_i, p, w)

    # Smoothed values = first row of coeffs dotted with y window
    y_win               = y[win_idx]                            # (n_i, window)
    y_smoothed[idx]     = (coeffs[:, 0, :] * y_win).sum(axis=1)

    # Border extrapolation — use polynomial from first / last interior window
    first_coeffs = coeffs[0] @ y_win[0]                        # (poly_order,)
    last_coeffs  = coeffs[-1] @ y_win[-1]                      # (poly_order,)

    # Left border
    li      = np.arange(half_window)
    x_li    = (x[li] - x[half_window])[:, None] ** powers[None, :]
    y_smoothed[li] = x_li @ first_coeffs

    # Right border
    ri      = np.arange(n - half_window, n)
    x_ri    = (x[ri] - x[-half_window - 1])[:, None] ** powers[None, :]
    y_smoothed[ri] = x_ri @ last_coeffs

    return y_smoothed


# ── 2. double_savgol — unchanged logic, faster non_uniform_savgol underneath ──

def double_savgol_original(ts, double=True, window1_min_obs=11, window1_max=21,
                  window2=59, polynom1=3, polynom2=3, limit=61):
    ts_tmp  = ts.copy()
    n_valid = int(np.sum(~np.isnan(ts_tmp)))
    window1 = int(np.clip(
        int(n_valid / 4) // 2 * 2 + 1,
        7, window1_max
    ))

    if double and n_valid > window1_min_obs:
        mask = ~np.isnan(ts_tmp)
        ts_tmp[mask] = non_uniform_savgol(
            np.where(mask)[0].astype(float),
            ts_tmp[mask],
            window=window1, polynom=polynom1
        )

    ts_interp = pd.Series(ts_tmp)
    ts_interp = ts_interp.interpolate(method='linear', limit_area='inside', limit=limit)
    ts_interp = ts_interp.interpolate(method='linear', limit=None,
                                      limit_direction='both', limit_area='outside')

    if window2 < ts_interp.size:
        try:
            ts_smooth = savgol_filter(ts_interp, window_length=window2, polyorder=polynom2)
        except np.linalg.LinAlgError:
            ts_smooth = ts_interp.rolling(21, center=True).mean().values
    else:
        print('RuntimeWarning: window2 >= ts length. Returning 21-pt rolling mean.')
        ts_smooth = ts_interp.rolling(21, center=True).mean().values

    return ts_smooth

# trying faster non-pandas approach
def linear_interp_numpy(ts, limit=61):
    out  = ts.copy()
    nans = np.isnan(out)
    
    # No NaNs — nothing to do
    if not nans.any():
        return out
    
    # All NaNs — nothing to interpolate
    if nans.all():
        return out

    x     = np.arange(len(out))
    valid = ~nans

    first_valid = x[valid][0]
    last_valid  = x[valid][-1]
    inside      = (x >= first_valid) & (x <= last_valid)

    filled = np.interp(x, x[valid], out[valid])

    if limit is not None:
        last_valid_idx = np.maximum.accumulate(np.where(~nans, x, -1))
        run_lengths    = np.where(nans, x - last_valid_idx, 0)
        within_limit   = run_lengths <= limit
        fill_mask      = inside & nans & within_limit
    else:
        fill_mask = inside & nans

    out[fill_mask]  = filled[fill_mask]

    still_nan       = np.isnan(out)
    if still_nan.any():
        out[still_nan] = filled[still_nan]

    return out

def double_savgol(ts, double=True, window1_min_obs=11, window1_max=21,
                  window2=59, polynom1=3, polynom2=3, limit=61):
    if np.isnan(ts).all():
        return ts
    else:
        ts_tmp  = ts.copy()
        n_valid = int(np.sum(~np.isnan(ts_tmp)))
        window1 = int(np.clip(
            int(n_valid / 4) // 2 * 2 + 1,
            7, window1_max
        ))
    
        if double and n_valid > window1_min_obs:
            mask = ~np.isnan(ts_tmp)
            ts_tmp[mask] = non_uniform_savgol(
                np.where(mask)[0].astype(float),
                ts_tmp[mask],
                window=window1, polynom=polynom1
            )
    
        # Replace pandas interpolation with NumPy version
        ts_interp = linear_interp_numpy(ts_tmp, limit=limit)
    
        if window2 < len(ts_interp):
            try:
                ts_smooth = savgol_filter(ts_interp, window_length=window2,
                                          polyorder=polynom2)
            except np.linalg.LinAlgError:
                ts_smooth = np.convolve(ts_interp,
                                        np.ones(21)/21, mode='same')
        else:
            ts_smooth = np.convolve(ts_interp, np.ones(21)/21, mode='same')
    
        return ts_smooth


# ── 3. Parallel wrapper using joblib across (x, y) pixels ─────────────────────

def _process_chunk(pixels_yx, arr3d, kwargs):
    """
    Process a list of (iy, ix) pixel indices.
    arr3d : numpy array, shape (time, y, x)
    Returns list of (iy, ix, smoothed_1d).
    """
    results = []
    for iy, ix in pixels_yx:
        ts     = arr3d[:, iy, ix]
        smooth = double_savgol(ts, **kwargs)
        results.append((iy, ix, smooth))
    return results

def smooth_array_parallel(arr3d, kwargs, n_jobs=-1, chunk_size=500):
    """
    Parallelise double_savgol over all (y, x) pixels using joblib.

    Parameters
    ----------
    arr3d      : np.ndarray, shape (time, y, x)
    kwargs     : dict passed to double_savgol
    n_jobs     : number of parallel workers (-1 = all cores)
    chunk_size : pixels per job (tune for your HPC node)
    """
    nt, ny, nx  = arr3d.shape
    out         = np.full_like(arr3d, np.nan, dtype=float)

    # Build flat list of all pixel indices, then batch into chunks
    all_yx  = [(iy, ix) for iy in range(ny) for ix in range(nx)]
    batches = [all_yx[i:i + chunk_size] for i in range(0, len(all_yx), chunk_size)]

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(_process_chunk)(batch, arr3d, kwargs)
        for batch in batches
    )

    for batch_result in results:
        for iy, ix, smooth in batch_result:
            out[:, iy, ix] = smooth

    return out


# ── 4. High-level xarray wrapper ───────────────────────────────────────────────

def smooth_xr_parallel(dat, smooth_dict, n_jobs=-1, chunk_size=2000):
    """
    Drop-in replacement for smooth_xr that parallelises over pixels.

    Parameters
    ----------
    dat         : xr.DataArray with dims (time, y, x)
    smooth_dict : dict with keys smooth_window1_max, smooth_window2, smooth_limit
    n_jobs      : -1 → use all available cores
    chunk_size  : pixels per joblib batch
    """
    kwargs = dict(
        double        = True,
        window1_max   = smooth_dict['smooth_window1_max'],
        window2       = smooth_dict['smooth_window2'],
        limit         = smooth_dict['smooth_limit'],
    )

    # Bring data into memory as a single numpy array (time, y, x)
    print("Loading data into memory …")
    arr = dat.transpose('time', 'y', 'x').values   # triggers Dask compute if needed

    num_pixels = arr.shape[1] * arr.shape[2]
    num_tasks = int(np.ceil(num_pixels/chunk_size))
    print(f"Smoothing {num_pixels} pixels ({num_tasks} tasks) "
          f"on {multiprocessing.cpu_count()} logical cores (n_jobs={n_jobs}) …")

    arr_smooth = smooth_array_parallel(arr, kwargs,
                                       n_jobs=n_jobs, chunk_size=chunk_size)

    # Wrap back into an xarray DataArray with original coordinates
    dat_t = dat.transpose('time', 'y', 'x')
    out   = xr.DataArray(
        arr_smooth,
        coords = dat_t.coords,
        dims   = dat_t.dims,
        attrs  = dat.attrs,
    )
    return out.transpose('time', 'y', 'x')


# ── 5. Original apply_ufunc approach — kept for reference / Dask cluster use ───

def smooth_xr_dask(dat_chunked, dims, kwargs={'double': True}):
    """
    Original approach — useful when you have a distributed Dask cluster
    (e.g. dask-jobqueue on SLURM/PBS).  Chunk along x and y so Dask
    schedules one task per spatial tile instead of one task per pixel.
    """
    # Rechunk: keep full time axis per task, tile spatially
    #dat_chunked = dat.chunk({'time': -1, 'y': 50, 'x': 50})

    xr_smoothed = xr.apply_ufunc(
        double_savgol,
        dat_chunked,
        kwargs       = kwargs,
        input_core_dims  = [dims],
        output_core_dims = [dims],
        dask         = 'parallelized',
        vectorize    = True,
        output_dtypes= [float],
    )
    return xr_smoothed.transpose('time', 'y', 'x')