import numpy as np

wav_BLUE = 510 - (510 - 450) / 2
wav_GREEN = 590 - (590 - 530) / 2
wav_RED = 670 - (670 - 640) / 2
wav_NIR = 880 - (880 - 850) / 2
wav_SWIR1 = 1650 - (1650 - 1570) / 2
wav_SWIR2 = 2290 - (2290 - 2110) / 2


def ndvi_func(src):
    # Normalized Difference Vegetation Index
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    ndvi = (band_NIR - band_RED) / (band_NIR + band_RED)
    ndvi = ndvi.where((ndvi < 1.0) & (ndvi > -1.0))
    return ndvi


def dfi_func(src):
    # Dead Fuels Index
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    dfi = 100 * (1 - band_SWIR2 / band_SWIR1) * band_RED / band_NIR
    return dfi


def ndti_func(src):
    # Normalized Difference Tillage Index
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    ndti = (band_SWIR1 - band_SWIR2) / (band_SWIR1 + band_SWIR2)
    ndti = ndti.where((ndti < 1.0) & (ndti > -1.0))
    return ndti


def savi_func(src):
    # Soil Adjusted Vegetation Index
    band_RED = src['RED'].where(src['RED'] != -9999) * 0.0001
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999) * 0.0001
    savi = ((band_NIR - band_RED) /
             (band_NIR + band_RED + 0.5)) * (1 + 0.5)
    return savi


def satvi_func(src):
    # Soil Adjustted Total Vegetation Index
    band_RED = src['RED'].where(src['RED'] != -9999) * 0.0001
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999) * 0.0001
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999) * 0.0001
    satvi = ((band_SWIR1 - band_RED) /
             (band_SWIR1 + band_RED + 0.5)) * (1 + 0.5) - (band_SWIR2 / 2.0)
    return satvi


def ndii7_func(src):
    # Normalized Difference Infrared Index 7 or Normalized Burn Ratio
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    ndii7 = (band_NIR - band_SWIR2) / (band_NIR + band_SWIR2)
    return ndii7


def ndwi_func(src):
    # Normalized Difference Water Index
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999)
    ndwi = (band_NIR - band_SWIR1) / (band_NIR + band_SWIR1)
    return ndwi


def evi_func(src):
    # Enhanced Vegetation Index
    band_BLUE = src['BLUE'].where(src['BLUE'] != -9999) * 0.0001
    band_RED = src['RED'].where(src['RED'] != -9999) * 0.0001
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999) * 0.0001
    evi = 2.5 * ((band_NIR - band_RED) / (band_NIR + 6*band_RED - 7.5*band_BLUE + 1))
    return evi

def tcbi_func(src):
    # Tassled Cap Brightness Index
    band_BLUE = src['BLUE'].where(src['BLUE'] != -9999) * 0.0001
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999) * 0.0001
    band_RED = src['RED'].where(src['RED'] != -9999) * 0.0001
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999) * 0.0001
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999) * 0.0001
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999) * 0.0001
    tcbi = 0.2043*band_BLUE + 0.4158*band_GREEN + 0.5524*band_RED + 0.5741*band_NIR + 0.3124*band_SWIR1 + 0.2303*band_SWIR2
    return tcbi


def tcgi_func(src):
    # Tassled Cap Greenness Index
    band_BLUE = src['BLUE'].where(src['BLUE'] != -9999) * 0.0001
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999) * 0.0001
    band_RED = src['RED'].where(src['RED'] != -9999) * 0.0001
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999) * 0.0001
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999) * 0.0001
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999) * 0.0001
    tcgi = -0.1603*band_BLUE - 0.2819*band_GREEN - 0.4934*band_RED + 0.7940*band_NIR - 0.0002*band_SWIR1 - 0.1446*band_SWIR2
    return tcgi


def tcwi_func(src):
    # Tassled Cap Wetness Index
    band_BLUE = src['BLUE'].where(src['BLUE'] != -9999) * 0.0001
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999) * 0.0001
    band_RED = src['RED'].where(src['RED'] != -9999) * 0.0001
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999) * 0.0001
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999) * 0.0001
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999) * 0.0001
    tcwi = 0.0315*band_BLUE + 0.2021*band_GREEN + 0.3102*band_RED + 0.1594*band_NIR - 0.6806*band_SWIR1 -0.6109*band_SWIR2
    return tcwi


def psri_func(src):
    # Plant Senesence Reflectance Index
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999)
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    psri = (band_RED - band_GREEN) / band_NIR
    return psri


def mtvi1_func(src):
    # Modified Triangular Vegetation Index 1
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999) * 0.0001
    band_RED = src['RED'].where(src['RED'] != -9999) * 0.0001
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999) * 0.0001
    mtvi1 = 1.2*(1.2*(band_NIR - band_GREEN) - 2.5*(band_RED - band_GREEN))
    return mtvi1


def nci_func(src):
    # Normalized Canopy Index
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999)
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999)
    nci = (band_SWIR1 - band_GREEN) / (band_SWIR1 + band_GREEN)
    return nci


def ndci_func(src):
    # Normalized Difference Cover Index or Normalized Difference Senescent Vegetation Index
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_SWIR1 = src['SWIR1'].where(src['SWIR1'] != -9999)
    ndci = (band_SWIR1 - band_RED) / (band_SWIR1 + band_RED)
    return ndci


def rdvi_func(src):
    # Renormalized Difference Vegetation Index
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    rdvi = (band_NIR - band_RED) / np.sqrt(band_NIR + band_RED)
    return rdvi
    

def bai_126_func(src):
    band_BLUE = src['BLUE'].where(src['BLUE'] != -9999)
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    bai_126_x1 = (wav_GREEN - wav_BLUE) / 2500
    bai_126_y1 = band_GREEN * 0.0001 - band_BLUE * 0.0001
    bai_126_x2 = (wav_SWIR2 - wav_GREEN) / 2500
    bai_126_y2 = band_SWIR2 * 0.0001 - band_GREEN * 0.0001
    bai_126 = 180 - np.arctan(bai_126_y1 / bai_126_x1) * 57.2958 + np.arctan(bai_126_y2 / bai_126_x2) * 57.2958
    bai_126 = bai_126.where(bai_126 > 0)
    return bai_126


def bai_136_func(src):
    band_BLUE = src['BLUE'].where(src['BLUE'] != -9999)
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    bai_136_x1 = (wav_RED - wav_BLUE) / 2500
    bai_136_y1 = band_RED * 0.0001 - band_BLUE * 0.0001
    bai_136_x2 = (wav_SWIR2 - wav_RED) / 2500
    bai_136_y2 = band_SWIR2 * 0.0001 - band_RED * 0.0001
    bai_136 = 180 - np.arctan(bai_136_y1 / bai_136_x1) * 57.2958 + np.arctan(bai_136_y2 / bai_136_x2) * 57.2958
    bai_136 = bai_136.where(bai_136 > 0)
    return bai_136


def bai_146_func(src):
    band_BLUE = src['BLUE'].where(src['BLUE'] != -9999)
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    bai_146_x1 = (wav_NIR - wav_BLUE) / 2500
    bai_146_y1 = band_NIR * 0.0001 - band_BLUE * 0.0001
    bai_146_x2 = (wav_SWIR2 - wav_NIR) / 2500
    bai_146_y2 = band_SWIR2 * 0.0001 - band_NIR * 0.0001
    bai_146 = 180 - np.arctan(bai_146_y1 / bai_146_x1) * 57.2958 + np.arctan(bai_146_y2 / bai_146_x2) * 57.2958
    bai_146 = bai_146.where(bai_146 > 0)
    return bai_146


def bai_236_func(src):
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999)
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    bai_236_x1 = (wav_RED - wav_GREEN) / 2500
    bai_236_y1 = band_RED * 0.0001 - band_GREEN * 0.0001
    bai_236_x2 = (wav_SWIR2 - wav_RED) / 2500
    bai_236_y2 = band_SWIR2 * 0.0001 - band_RED * 0.0001
    bai_236 = 180 - np.arctan(bai_236_y1 / bai_236_x1) * 57.2958 + np.arctan(bai_236_y2 / bai_236_x2) * 57.2958
    bai_236 = bai_236.where(bai_236 > 0)
    return bai_236


def bai_246_func(src):
    band_GREEN = src['GREEN'].where(src['GREEN'] != -9999)
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    bai_246_x1 = (wav_NIR - wav_GREEN) / 2500
    bai_246_y1 = band_NIR * 0.0001 - band_GREEN * 0.0001
    bai_246_x2 = (wav_SWIR2 - wav_NIR) / 2500
    bai_246_y2 = band_SWIR2 * 0.0001 - band_NIR * 0.0001
    bai_246 = 180 - np.arctan(bai_246_y1 / bai_246_x1) * 57.2958 + np.arctan(bai_246_y2 / bai_246_x2) * 57.2958
    bai_246 = bai_246.where(bai_246 > 0)
    return bai_246


def bai_346_func(src):
    band_RED = src['RED'].where(src['RED'] != -9999)
    band_NIR = src['NIR1'].where(src['NIR1'] != -9999)
    band_SWIR2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    bai_346_x1 = (wav_NIR - wav_RED) / 2500
    bai_346_y1 = band_NIR * 0.0001 - band_RED * 0.0001
    bai_346_x2 = (wav_SWIR2 - wav_NIR) / 2500
    bai_346_y2 = band_SWIR2 * 0.0001 - band_NIR * 0.0001
    bai_346 = 180 - np.arctan(bai_346_y1 / bai_346_x1) * 57.2958 + np.arctan(bai_346_y2 / bai_346_x2) * 57.2958
    bai_346 = bai_346.where(bai_346 > 0)
    return bai_346

