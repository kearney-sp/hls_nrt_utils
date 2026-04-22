import pickle
import numpy as np
from pkg_resources import resource_filename

model_dict = {
    'cper_biomass': resource_filename('hlsstack', 'models/CPER_HLS_to_VOR_biomass_model_pls_20260417.pk'),
    'cper_cover': resource_filename('hlsstack', 'models/CPER_HLS_to_LPI_cover_model_20260417.pk'),
    'cper_cp': resource_filename('hlsstack', 'models/CPER_HLS_cp_ndvi_2014_2023_04222026.pk'),
    'tbng_biomass': resource_filename('hlsstack', 'models/TB_HLS_to_VOR_biomass_model_pls_20260417.pk'),
    'cper_cover_old': resource_filename('hlsstack', 'models/CPER_HLS_to_LPI_cover_pls_binned_model.pk'),
    'cper_biomass_old_lm': resource_filename('hlsstack', 'models/CPER_HLS_to_VOR_biomass_model_lr_simp.pk'),
    'cper_biomass_old_pls': resource_filename('hlsstack', 'models/CPER_HLS_to_VOR_biomass_model_pls_20241015.pk')
}

def load_model(model_name):
   model = pickle.load(open(model_dict[model_name], 'rb'))
   return model

def xfrm_y(y):
    x = y.copy()
    x[np.where(x > 0)] = np.sqrt(x[np.where(x > 0)])
    x[np.where(x <= 0)] = 0.0
    return x

def bxfrm_y(y):
    x = y.copy()
    x[np.where(x > 0)] = x[np.where(x > 0)]**2
    x[np.where(x <= 0)] = 0.0
    return x