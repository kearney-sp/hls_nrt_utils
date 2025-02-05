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
from sklearn.exceptions import InconsistentVersionWarning

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


def predict_biomass(dat, model, se=True):
    """ Predict biomass (kg/ha) and standard error of prediction from existing linear model
        dat (xarray dataset) = new data in xarray Dataset format
        model (object) = opened existing model using pickle
        se (boolean) """

    model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]

    new_df = pd.DataFrame()
    for v in model_vars:
        new_df[v] = func_dict[v](dat).values.flatten()
    new_df['bm'] = np.exp(model.predict(new_df))

    if se:
        new_df.loc[~new_df.bm.isnull(), 'bm_se_log'] = model.get_prediction(new_df.loc[~new_df.bm.isnull()]).se_obs
        return [xr.DataArray(data=new_df['bm'].values.reshape(dat[list(dat.keys())[0]].shape),
                        coords=dat.coords),
                xr.DataArray(data=new_df['bm'].values.reshape(dat[list(dat.keys())[0]].shape),
                             coords=dat.coords)]
    else:
        return xr.DataArray(data=new_df['bm'].values.reshape(dat[list(dat.keys())[0]].shape),
                        coords=dat.coords)


def pred_bm(dat, model):
    #model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]
    model_vars = model.feature_names_in_
    dat_masked = dat.where(dat.notnull())

    def pred_func(*args, mod_vars_np):
        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
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
    mod_list = glob.glob(os.path.join(mod_boot_dir,'*.pk'))
    model_vars = model.feature_names_in_
    dat_masked = dat.where(dat.notnull)
    
    def pred_func(*args, mod_vars_np):
        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
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


def pred_bm2(dat, model):
    model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]

    dat_masked = dat.where(dat.notnull)

    #dims_list = [[dim] for v in model_vars]

    def pred_func(*args, mod_vars_np):
        vars_dict_np = {}
        for idx, v in enumerate(mod_vars_np):
            vars_dict_np[v] = args[idx]
        #print(vars_dict_np)
        bm_np = np.ones_like(args[0]) * np.nan
        mask = np.any(np.isnan(args), axis=0)
        bm_np[~mask] = np.exp(model.predict(vars_dict_np))
        #print(bm_np)
        return bm_np.astype('int16')

    def pred_func_xr(dat_xr, model_vars_xr):
        dims = [['z'] for v in model_vars]
        #dat_xr = dat_xr.stack(z=('y', 'x'))
        vars_list_xr = []
        for v in model_vars_xr:
            vars_list_xr.append(func_dict[v](dat_xr).stack(z=('y', 'x')))
        bm_xr = xr.apply_ufunc(pred_func,
                               *vars_list_xr,
                               kwargs=dict(mod_vars_np=np.array(model_vars_xr)),
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=dims,
                               output_core_dims=[dims[0]],
                               output_dtypes=['int16'])
        return bm_xr.unstack('z')

    #bm_out = pred_func_xr(dat_masked, model_vars, dims_list)
    bm_out = dat_masked.map_blocks(pred_func_xr, kwargs=dict(model_vars_xr=model_vars), template=dat_masked['RED'])

    return bm_out


def pred_bm_se2(dat, model):
    model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]

    dat_masked = dat.where(dat.notnull)

    #dims_list = [[dim] for v in model_vars]

    def pred_func(*args, mod_vars_np):
        mask = np.any(np.isnan(args), axis=0)
        vars_dict_np = {}
        for idx, v in enumerate(mod_vars_np):
            vars_dict_np[v] = args[idx]
        #print(vars_dict_np)
        se_np = np.ones_like(args[0]) * np.nan
        se_np[~mask] = model.get_prediction(vars_dict_np).se_obs
        #print(bm_np)
        return se_np.astype('float32')

    def pred_func_xr(dat_xr, model_vars_xr):
        dims = [['z'] for v in model_vars]
        #dat_xr = dat_xr.stack(z=('y', 'x'))
        vars_list_xr = []
        for v in model_vars_xr:
            vars_list_xr.append(func_dict[v](dat_xr).stack(z=('y', 'x')))
        bm_xr = xr.apply_ufunc(pred_func,
                               *vars_list_xr,
                               kwargs=dict(mod_vars_np=np.array(model_vars_xr)),
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=dims,
                               output_core_dims=[dims[0]],
                               output_dtypes=['float32'])
        return bm_xr.unstack('z')

    #bm_out = pred_func_xr(dat_masked, model_vars, dims_list)
    se_out = dat_masked.map_blocks(pred_func_xr, kwargs=dict(model_vars_xr=model_vars), template=dat_masked['RED'])

    return se_out


def pred_bm_thresh2(dat, model, thresh_kg):
    model_vars = [n for n in model.params.index if ":" not in n and "Intercept" not in n]

    dat_masked = dat.where(dat.notnull)

    #dims_list = [[dim] for v in model_vars]

    def pred_func(*args, mod_vars_np):
        mask = np.any(np.isnan(args), axis=0)
        vars_dict_np = {}
        for idx, v in enumerate(mod_vars_np):
            vars_dict_np[v] = args[idx]
        #print(vars_dict_np)
        bm_np = model.predict(vars_dict_np)
        se_np = model.get_prediction(vars_dict_np).se_obs
        thresh_pre = (np.log(thresh_kg) - bm_np) / se_np
        thresh_np = np.ones_like(args[0]) * np.nan
        thresh_np[~mask] = st.norm.cdf(thresh_pre)
        #print(bm_np)
        return thresh_np.astype('float32')

    def pred_func_xr(dat_xr, model_vars_xr):
        dims = [['z'] for v in model_vars]
        #dat_xr = dat_xr.stack(z=('y', 'x'))
        vars_list_xr = []
        for v in model_vars_xr:
            vars_list_xr.append(func_dict[v](dat_xr).stack(z=('y', 'x')))
        bm_xr = xr.apply_ufunc(pred_func,
                               *vars_list_xr,
                               kwargs=dict(mod_vars_np=np.array(model_vars_xr)),
                               dask='parallelized',
                               vectorize=True,
                               input_core_dims=dims,
                               output_core_dims=[dims[0]],
                               output_dtypes=['float32'])
        return bm_xr.unstack('z')

    #bm_out = pred_func_xr(dat_masked, model_vars, dims_list)
    se_out = dat_masked.map_blocks(pred_func_xr, kwargs=dict(model_vars_xr=model_vars), template=dat_masked['RED'])

    return se_out


def pred_cov_sma(dat, ends_dict):
    end_classes = list(ends_dict.keys())
    end_vars = list(ends_dict[end_classes[0]].keys())
    end_vals = np.array([list(ends_dict[c].values()) for c in end_classes])

    def pred_unmix(*args, ends, idx):
        mat = np.array(args).T
        unmixed = amaps.UCLS(mat, np.array(ends[0]))
        unmixed[unmixed < 0] = 0
        unmixed[unmixed > 1.0] = 1.0
        return unmixed[:, idx]

    def pred_unmix_xr(dat_xr, ends, idx, name):
        dat_xr = dat_xr.stack(z=('y', 'x'))
        dims_list = [['z'] for c in end_vars]
        vars_list_xr = []
        for v in end_vars:
            vars_list_xr.append(func_dict[v](dat_xr))
        unmixed_xr = xr.apply_ufunc(pred_unmix,
                                    *vars_list_xr,
                                    dask='parallelized',
                                    vectorize=True,
                                    input_core_dims=dims_list,
                                    output_core_dims=[dims_list[0]],
                                    output_dtypes=['float32'],
                                    kwargs=dict(ends=ends, idx=idx))
        unmixed_xr = unmixed_xr.assign_coords(type=name)
        return unmixed_xr.unstack('z')

    covArrays = []
    for idx, c in enumerate(end_classes):
        covArrays.append(pred_unmix_xr(dat, ends=[end_vals], idx=idx, name=c))

    dat_cov = xr.concat(covArrays, dim='type', join='override', combine_attrs='drop')
    #dat_cov['type'] = [c for c in end_classes]
    dat_cov = dat_cov.to_dataset(dim='type')
    return dat_cov






def pred_cov_sma2(dat, ends_dict):
    end_classes = list(ends_dict.keys())
    end_vars = list(ends_dict[end_classes[0]].keys())
    end_vals = np.array([list(ends_dict[c].values()) for c in end_classes])

    def pred_unmix(*args, ends, idx):
        mat = np.array(args).T
        unmixed = amaps.UCLS(mat, np.array(ends[0]))
        unmixed[unmixed < 0] = 0
        unmixed[unmixed > 1.0] = 1.0
        return unmixed[:, idx]

    def pred_unmix_xr(dat_xr, ends, idx, name):
        dims = [['z'] for c in end_vars]
        vars_list_xr = []
        for v in end_vars:
            vars_list_xr.append(func_dict[v](dat_xr).stack(z=('y', 'x')))
        unmixed_xr = xr.apply_ufunc(pred_unmix,
                                    *vars_list_xr,
                                    dask='parallelized',
                                    vectorize=True,
                                    input_core_dims=dims,
                                    output_core_dims=[dims[0]],
                                    output_dtypes=['float32'],
                                    kwargs=dict(ends=ends, idx=idx))
        #unmixed_xr = unmixed_xr.assign_coords(type=name)
        return unmixed_xr.unstack('z')

    covArrays = []
    for idx, c in enumerate(end_classes):
        covArrays.append(dat.map_blocks(pred_unmix_xr, kwargs=dict(ends=[end_vals], idx=idx, name=c),
                                        template=dat['RED']).assign_coords(type=c))

    dat_cov = xr.concat(covArrays, dim='type', join='override', combine_attrs='drop')
    #dat_cov['type'] = [c for c in end_classes]
    dat_cov = dat_cov.to_dataset(dim='type')
    return dat_cov
