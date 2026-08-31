#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='hls_nrt_utils',
      version='0.1.1',
      description='PIP-installable utilities for performing near-real-time analyses using Harmonized Landsat-Sentinel dataset.',
      author='Sean Patrick Kearney',
      author_email='sean.patrick@hotmail.com',
      license='MIT',
      packages=['hlsstack', 'hlsstack.hls_funcs', 'hlsstack.models', 'hlsstack.objects', 'hlsstack.utils'],
      install_requires=['cartopy','certifi','cvxopt','dask','geopandas','joblib','numpy','pandas','rasterio','rioxarray','scikit-image','scikit-learn','scipy','stackstac','statsmodels','xarray','pysptools'],
      package_data={'hlsstack.models': ['*.pk', '*.tif']},
      include_package_data=True
    )
