import pickle

model_dict = {
              'cper_biomass': './CPER_HLS_to_VOR_biomass_model_lr_simp.pk',
              'cper_cover': './CPER_HLS_to_LPI_cover_pls_binned_model.pk'
              }

def load_model(model_name):
   model = pickle.load(open(model_dict[model_name], 'rb'))
   return model
