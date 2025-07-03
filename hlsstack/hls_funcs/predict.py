import pickle
import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
import random
from hlsstack.hls_funcs.bands import *
from hlsstack.hls_funcs.indices import *
from pysptools.abundance_maps import amaps
import scipy.stats as st
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
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
    #model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]
    model_vars = model.feature_names_in_
    dat_masked = dat.where(dat.notnull())

    def pred_func(*args, mod_vars_np):
        #warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        vars_dict_np = {}
        for idx, v in enumerate(mod_vars_np):
            vars_dict_np[v] = args[idx]
        df_vars = pd.DataFrame(vars_dict_np, columns=mod_vars_np)
        bm_np = np.ones_like(args[0]) * np.nan
        mask = np.any(np.isnan(args), axis=0)
        if len(df_vars[model.feature_names_in_].dropna(how='any')) > 0:
            bm_np[~mask] = model.predict(df_vars[model.feature_names_in_].dropna(how='any'))
        return bm_np.astype('int16')

    def pred_func_xr(dat_xr, model_vars_xr):
        dat_xr = dat_xr.stack(z=('y', 'x')).persist()
        dims_list = [['z'] for v in model_vars_xr]
        vars_list_xr = []
        for v in model_vars_xr:
            vars_list_xr.append(func_dict[v](dat_xr))
        bm_xr = xr.apply_ufunc(pred_func,
                               *vars_list_xr,
                               kwargs=dict(mod_vars_np=np.array(model_vars_xr)),
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=dims_list,
                               output_core_dims=[dims_list[0]],
                               output_dtypes=['int16'])
        return bm_xr.unstack('z')

    bm_out = pred_func_xr(dat_masked, model_vars)

    return bm_out


def pred_bm_se(dat, model, mod_boot_dir, nboot=100, avg_std=145.21):
    # see https://doi.org/10.1016/j.jbusres.2016.03.049
    mod_list = glob.glob(os.path.join(mod_boot_dir,'*.pk'))
    model_vars = model.feature_names_in_
    dat_masked = dat.where(dat.notnull)
    
    def pred_func(*args, mod_vars_np):
        #warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        vars_dict_np = {}
        for idx, v in enumerate(mod_vars_np):
            vars_dict_np[v] = args[idx]
        df_vars = pd.DataFrame(vars_dict_np, columns=mod_vars_np)
        se_np = np.ones_like(args[0]) * np.nan
        mask = np.any(np.isnan(args), axis=0)
        if len(df_vars[model.feature_names_in_].dropna(how='any')) > 0:
            rand_mod_idx = random.sample(range(len(mod_list)), nboot)
            preds = []
            for b in rand_mod_idx:
                with open(mod_list[b], 'rb') as f:
                    mod_tmp = pd.compat.pickle_compat.load(f)
                preds_tmp = mod_tmp.predict(df_vars[mod_tmp.feature_names_in_].dropna(how='any'))
                preds.append(pd.Series(preds_tmp, name='predy_' + str(b)))
            df_preds = pd.concat(preds, axis=1)
            se_np[~mask] = df_preds.std(axis=1).values + avg_std
        return se_np.astype('float32')

    def pred_func_xr(dat_xr, model_vars_xr):
        dat_xr = dat_xr.stack(z=('y', 'x'))
        dims_list = [['z'] for v in model_vars_xr]
        vars_list_xr = []
        for v in model_vars_xr:
            vars_list_xr.append(func_dict[v](dat_xr))
        se_xr = xr.apply_ufunc(pred_func,
                               *vars_list_xr,
                               kwargs=dict(mod_vars_np=np.array(model_vars_xr)),
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=dims_list,
                               output_core_dims=[dims_list[0]],
                               output_dtypes=['float32'])
        return se_xr.unstack('z')

    se_out = pred_func_xr(dat_masked, model_vars)

    return se_out


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
    pls2_mod = model

    band_list = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2',
                 'DFI', 'NDVI', 'NDTI', 'SATVI', 'NDII7',
                 'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346']

    def pred_cov_np(*args):
        mat = np.array(args).T
        unmixed = np.ones((mat.shape[0], 4)) * np.nan
        if mat[~np.any(np.isnan(mat), axis=1), :].shape[0] > 0:
            mat2 = PolynomialFeatures(2).fit_transform(mat[~np.any(np.isnan(mat), axis=1), :])
            unmixed[~np.any(np.isnan(mat), axis=1), :] = pls2_mod.predict(mat2)
            unmixed[np.where(unmixed < 0)] = 0
            unmixed[np.where(unmixed > 1)] = 1
        return unmixed[:, 0], unmixed[:, 1], unmixed[:, 2], unmixed[:, 3]

    def pred_cov_xr(dat_xr, name):
        dat_xr = dat_xr.stack(z=('y', 'x'))
        vars_list_xr = []
        for v in band_list:
            vars_list_xr.append(func_dict[v](dat_xr))
        unmixed_xr = xr.apply_ufunc(pred_cov_np,
                                    *vars_list_xr,
                                    dask='parallelized',
                                    vectorize=True,
                                    input_core_dims=np.repeat(['z'], len(band_list)),
                                    output_core_dims=['z', 'z', 'z', 'z'],
                                    output_dtypes=['float32', 'float32', 'float32', 'float32'])
        cov_xr = xr.concat(unmixed_xr, dim='type').unstack('z')
        cov_xr = cov_xr.assign_coords(type=name)
        return cov_xr.to_dataset(dim='type')

    dat_cov = pred_cov_xr(dat, name=['BARE', 'SD', 'GREEN', 'LITT'])
    return dat_cov


def pred_cp(dat, model):
    dat_masked = dat.where(dat.notnull())

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
            except IndexError:
                return np.ones_like(ndvi_ts_mean) * np.nan
        else:
            return np.ones_like(ndvi_ts_mean) * np.nan
    
    def pred_func(ndvi_ts):
        if np.all(np.isnan(ndvi_ts)):
            cp_out = np.ones_like(ndvi_ts) * np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
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
                               *[dat_xr],
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=[['time']],
                               output_core_dims=[['time']],
                               output_dtypes=['float32'],
                              )
        return cp_xr

    cp_out_xr = pred_func_xr(dat_masked)

    return cp_out_xr