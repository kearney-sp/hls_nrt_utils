import requests as r
import os
from netrc import netrc
from subprocess import Popen
import stackstac
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import certifi
from pyproj import Transformer

# Dictionary (i.e., look-up table; LUT) of the HLS product bands mapped to names
lut = {'HLSS30':
       {'B01': 'COASTAL-AEROSOL',
        'B02': 'BLUE', 
        'B03': 'GREEN', 
        'B04': 'RED', 
        'B05': 'RED-EDGE1',
        'B06': 'RED-EDGE2', 
        'B07': 'RED-EDGE3',
        'B08': 'NIR-Broad',
        'B8A': 'NIR1', 
        'B09': 'WATER-VAPOR',
        'B10': 'CIRRUS',
        'B11': 'SWIR1', 
        'B12': 'SWIR2', 
        'Fmask': 'FMASK',
        'SZA': 'SZA',
        'SAA': 'SAA',
        'VZA': 'VZA',
        'VAA': 'VAA'},
       'HLSL30': 
       {'B01': 'COASTAL-AEROSOL',
        'B02': 'BLUE', 
        'B03': 'GREEN', 
        'B04': 'RED', 
        'B05': 'NIR1',
        'B06': 'SWIR1',
        'B07': 'SWIR2', 
        'B09': 'CIRRUS', 
        'B10': 'TIR1', 
        'B11': 'TIR2', 
        'Fmask': 'FMASK',
        'SZA': 'SZA',
        'SAA': 'SAA',
        'VZA': 'VZA',
        'VAA': 'VAA'}}

# List of all available/acceptable band names
all_bands = ['ALL',
             'COASTAL-AEROSOL',
             'BLUE',
             'GREEN',
             'RED',
             'RED-EDGE1',
             'RED-EDGE2',
             'RED-EDGE3',
             'NIR1',
             'SWIR1',
             'SWIR2',
             'CIRRUS',
             'TIR1',
             'TIR2',
             'WATER-VAPOR',
             'FMASK']

# list of just the bands currently used in functions
needed_bands = ['BLUE',
                'GREEN',
                'RED',
                'NIR1',
                'SWIR1',
                'SWIR2',
                'FMASK',
                'SZA',
                'SAA',
                'VZA',
                'VAA']


def HLS_CMR_STAC(hls_data, bbox_latlon, lim=100, aws=False, debug=False):
    """
    Define and execute a url-query of LPCLOUD DAAC for the HLSS30 and HLSL30 products.
    
    Parameters
    ----------
    hls_data: dict
        A dictionary with a 'data_range' key that is a 1-d array or tuple of length=2 with a 
        start and end date of the search query, written as strings ('YYYY-MM-DD')
    bbox_latlong: array
        A 1-d array of length=4 that is the bounding box (min_x, min_y, max_x, max_y), in
        latitude/longitude format, of the search query
    lim: int
        The maximum number of results to be returned by the search query
    aws: bool
        Is the search being done from an AWS server (default is False; True currently not working)
    debug: bool
        If True, print the search query url(s) (default is false)
        
    Returns
    -------
    dict
        A dictionary with two keys, one for each collection, whose values are lists of the urls to each 
        feature result of the query. The features are uls of individual images, each of which is an 
        individual band, for indivudal dates, for each collection.
    """

    # define the base urls for each collection
    lp_search_s30 = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search?&collections=HLSS30.v2.0'
    lp_search_l30 = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search?&collections=HLSL30.v2.0'
    
    # create a string for the bounding box from the defined bounds
    bbox = f'{bbox_latlon[0]},{bbox_latlon[1]},{bbox_latlon[2]},{bbox_latlon[3]}'  
    
    # create a string for the date range from defined date range
    date_time = hls_data['date_range'][0] + 'T00:00:00Z' + '/' + hls_data['date_range'][1] + 'T00:00:00Z'  
    
    # add in a limit parameter to retrieve 100 items at a time (will perform multiple searches if necessary)
    search_query_s30 = f"{lp_search_s30}&limit=100"   
    search_query_l30 = f"{lp_search_l30}&limit=100" 
    
    # add the bbox string to query    
    search_query2_s30 = f"{search_query_s30}&bbox={bbox}"   
    search_query2_l30 = f"{search_query_l30}&bbox={bbox}" 
    
    # add date range to query that already includes bbox
    search_query3_s30 = f"{search_query2_s30}&datetime={date_time}"  
    search_query3_l30 = f"{search_query2_l30}&datetime={date_time}"

    # create empty lists to store results
    s30_items = list()
    l30_items = list()
    
    if debug:
        # print the queries if debugging
        print(search_query3_s30)
        print(search_query3_l30)
    if lim > 100:
        # repeat search multiple times, limiting each search to 100 results
        for i in range(int(np.ceil(lim/100))):
            if i > 10:
                print('WARNING: Fetching more than 1000 records, this may result in a very large dataset.')
            
            # get just the features from the current query
            features_s30 = r.get(search_query3_s30).json()['features']
            features_l30 = r.get(search_query3_l30).json()['features']  
            
            # append all features from current query to the running list
            s30_items = s30_items + [h for h in features_s30] 
            l30_items = l30_items + [h for h in features_l30]
            
            if (len(s30_items) > 0) and (len(l30_items) > 0):
                # get the date of the earliest image in the current query results from both collections
                start_time = str(
                    max(datetime.strptime(s30_items[-1]['properties']['datetime'].split('T')[0],
                                          '%Y-%m-%d'),
                        datetime.strptime(l30_items[-1]['properties']['datetime'].split('T')[0], 
                                          '%Y-%m-%d')).date() + timedelta(days=1))
            elif len(s30_items) > 0:
                # get the date of the earliest image in the current query results from the S30 collection
                start_time = str(
                    datetime.strptime(s30_items[-1]['properties']['datetime'].split('T')[0], 
                                      '%Y-%m-%d').date() + timedelta(days=1))
            elif len(l30_items) > 0:
                # get the date of the earliest image in the current query results from the L30 collection
                start_time = str(
                    datetime.strptime(l30_items[-1]['properties']['datetime'].split('T')[0], 
                                      '%Y-%m-%d').date() + timedelta(days=1))
            else:
                # stop searching if no results are found
                break 
            
            # update query with new start time 
            date_time = start_time + 'T00:00:00Z' + '/' + hls_data['date_range'][1] + 'T00:00:00Z'
            search_query3_s30 = f"{search_query2_s30}&datetime={date_time}"  
            search_query3_l30 = f"{search_query2_l30}&datetime={date_time}" 
            if debug:
                # print the current queries if debugging
                print(search_query3_s30)
                print(search_query3_l30)
            if (len(features_s30) + len(features_l30)) == 0:
                # stop searching if no results are found
                break 
    else:
        # use for a single query when lim <= 100
        
        # get just the features from the query
        features_s30 = r.get(search_query3_s30).json()['features']
        features_l30 = r.get(search_query3_l30).json()['features'] 
        
        # append all features from query to the running list
        s30_items = s30_items + [h for h in features_s30]  
        l30_items = l30_items + [h for h in features_l30]
    
    if aws:
        # change the query url to point to the AWS S3 bucket
        for stac in s30_items:
            for band in stac['assets']:
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace(
                    'https://lpdaac.earthdata.nasa.gov/lp-prod-protected', 
                    's3://lp-prod-protected')
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace(
                    'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected', 
                    's3://lp-prod-protected')
                
        for stac in l30_items:
            for band in stac['assets']:
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace(
                    'https://lpdaac.earthdata.nasa.gov/lp-prod-protected', 
                    's3://lp-prod-protected')
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace(
                    'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected', 
                    's3://lp-prod-protected')
    return {'S30': s30_items,
            'L30': l30_items}


def setup_netrc(creds, aws=False):
    """
    Setup the credentials for querying the LPCLOUD DAAC stac. Creates a .netrc file which contains the
    Earthdata username and password. Register at https://urs.earthdata.nasa.gov/
    Also see
    https://www.earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/earthdata-login
    
    Parameters
    ----------
    creds: 1-d array (str)
        A length=2 array of two strings, the first being the username and the second being the password
        for an Earthdata account.
        
    aws: bool
        Is the srcipt being run on an AWS server? (default is False)
    
    Returns
    -------
    dict
        Dictionary with tempoary 'secretAccessKey', 'accessKeyId', and 'sessionToken' for the S3 bucket
    
    """
    urs = 'urs.earthdata.nasa.gov' 
    try:
        netrcDir = os.path.expanduser("~/.netrc")
        netrc(netrcDir).authenticators(urs)[0]
        del netrcDir

    # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
    except FileNotFoundError:
        homeDir = os.path.expanduser("~")
        Popen('touch {0}.netrc | chmod og-rw {0}.netrc | echo machine {1} >> {0}.netrc'.format(
            homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(creds[0], homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(creds[1], homeDir + os.sep), shell=True)
        del homeDir

    # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
    except TypeError:
        homeDir = os.path.expanduser("~")
        Popen('echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(creds[0], homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(creds[1], homeDir + os.sep), shell=True)
        del homeDir
    del urs
    if aws:
        return(r.get('https://lpdaac.earthdata.nasa.gov/s3credentials').json())
    else:
        return('')

    
def build_xr(stac_dict, lut=lut, bbox=None, stack_chunks=(3660, 3660), proj_epsg=32613):
    """
    Build an xarray.DataSet from a list of urls for individual HLS images using stackstac.
    
    Parameters
    ----------
    stac_dict: dict
        A dictionary with two keys, one for each collection, whose values are lists of the urls to each 
        feature result of the query. The features are uls of individual images, each of which is an 
        individual band, for indivudal dates, for each collection. (output from HLS_CMR_STAC)
        
    lut: dict
        A dictionary that maps all the HLS bands to common names used for subsequent band mapping.
        (default is the 'lut' dictionary defined above)
    
    bbox: tuple, optional (passed to stackstac.stack)
        Output spatial bounding-box, as a tuple of (min_x, min_y, max_x, max_y). This defines the (west, south,
        east, north) rectangle the output array will cover. Values must be in the same coordinate reference system
        as epsg.
        If None (default), the bounding box of all the input items is automatically used. So in most cases, you can
        leave bounds as None. You’d only need to set it when you want to use a custom bounding box.
        When bounds is given, any assets that don’t overlap those bounds are dropped. (see stackstack.stack 
        docs for more information)
    
    stack_chunks: tuple, optional (passed to stackstac.stack)
        The chunksize to use for the Dask array. Default: 1024. Picking a good chunksize will have significant 
        effects on performance! (see stackstack.stack docs for more information)
        
    proj_espg: int
        Reproject into this coordinate reference system, as given by an EPSG code. If None, uses whatever 
        CRS is set on all the items. In this case, all Items/Assets must have the proj:epsg field, 
        and it must be the same value for all of them. (default is for UTM Zone 13)
    
    Returns
    -------
    xarray.Dataset
        xarray.DataSetof all the STAC items, reprojected to the same grid and stacked by time. The DataSet’s 
        dimensions will be ("time", "band", "y", "x"). It’s backed by a lazy Dask array, so you can manipulate it
        without touching any data. By default, datset is chunked on the 'time' dimension.
    
    """
    try:
        # create the initial xarray.DataArray
        s30_stack = stackstac.stack(stac_dict['S30'], epsg=proj_epsg, resolution=30,
                                    assets=[i for i in lut['HLSS30'] if lut['HLSS30'][i] in needed_bands],
                                    bounds=bbox, 
                                    chunksize=stack_chunks)
        # rename the bands using the look-up table dictionary
        s30_stack['band'] = [lut['HLSS30'][b] for b in s30_stack['band'].values]
        # get the datetime from the timestamp
        s30_stack['time'] = [datetime.fromtimestamp(t) for t in s30_stack.time.astype('int').values//1000000000]
        # convert datetime to date format
        s30_stack['time'] = s30_stack['time'].dt.date
        # convert to xarray.DataSet on the band dimension, thereby creating variables for each band
        s30_stack = s30_stack.to_dataset(dim='band').reset_coords(['end_datetime', 'start_datetime'], drop=True)
        
    except ValueError:
        # warn of ValueError, typically caused by no S30 image urls
        print('Warning: ValueError in S30 stacking.')
        # create an empty variable for merging
        s30_stack = None
    try:
        # create the initial xarray.DataArray
        l30_stack = stackstac.stack(stac_dict['L30'], epsg=proj_epsg, resolution=30, 
                                    assets=[i for i in lut['HLSL30'] if lut['HLSL30'][i] in needed_bands],
                                    bounds=bbox,
                                    chunksize=stack_chunks)
        # rename the bands using the look-up table dictionary
        l30_stack['band'] = [lut['HLSL30'][b] for b in l30_stack['band'].values]
        # get the datetime from the timestamp
        l30_stack['time'] = [datetime.fromtimestamp(t) for t in l30_stack.time.astype('int').values//1000000000]
        # convert datetime to date format
        l30_stack['time'] = l30_stack['time'].dt.date
        # convert to xarray.DataSet on the band dimension, thereby creating variables for each band
        l30_stack = l30_stack.to_dataset(dim='band').reset_coords(['end_datetime', 'start_datetime'], drop=True)
    except ValueError:
        # warn of ValueError, typically caused by no L30 image urls
        print('Warning: ValueError in L30 stacking.')
        # create an empty variable for merging
        l30_stack = None
    if s30_stack is not None and l30_stack is not None:
        # stack the two datasets together on the time dimension
        hls_stack = xr.concat([s30_stack, l30_stack], dim='time')
    elif s30_stack is not None:
        # only pass the S30 dataset, L30 does not exist
        hls_stack = s30_stack
    elif l30_stack is not None:
        # only pass the L30 dataset, S30 does not exist
        hls_stack = l30_stack
    else:
        # notify that no L30 or S30 data were found
        print('No data found for date range')
    return hls_stack.chunk({'time': 1, 'y': -1, 'x': -1})
    

def get_hls(hls_data={}, bbox=[517617.2187, 4514729.5, 527253.4091, 4524372.5], transform_bbox=True,
            lut=lut, lim=100, aws=False, stack_chunks=(3660, 3660), proj_epsg=32613, debug=False):   
    # run functions
    transformer = Transformer.from_crs('epsg:' + str(proj_epsg), 'epsg:4326')
    if transform_bbox:
        bbox_lon, bbox_lat = transformer.transform(bbox[[0, 2]], bbox[[1, 3]])
        bbox_latlon = list(np.array(list(map(list, zip(bbox_lat, bbox_lon)))).flatten())
    else:
        bbox_latlon = bbox
    catalog = HLS_CMR_STAC(hls_data, bbox_latlon, lim, aws, debug)
    da  = build_xr(catalog, lut, bbox, stack_chunks, proj_epsg)
    return da


def setup_env(aws=False, creds=[]):
    #define gdalenv
    if aws:
        #import boto3
        #import rasterio as rio
        #from rasterio.session import AWSSession
        # set up creds
        s3_cred = setup_netrc(creds, aws=aws)
        env = dict(GDAL_HTTP_MAX_RETRY='5',
                   GDAL_HTTP_RETRY_DELAY='2',
                   GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
                   #AWS_NO_SIGN_REQUEST='YES',
                   GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
                   GDAL_SWATH_SIZE='200000000',
                   VSI_CURL_CACHE_SIZE='200000000',
                   CPL_VSIL_CURL_ALLOWED_EXTENSIONS='TIF',
                   GDAL_HTTP_UNSAFESSL='YES',
                   GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                   GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'),
                   AWS_REGION='us-west-2',
                   AWS_SECRET_ACCESS_KEY=s3_cred['secretAccessKey'],
                   AWS_ACCESS_KEY_ID=s3_cred['accessKeyId'],
                   AWS_SESSION_TOKEN=s3_cred['sessionToken'],
                   AWS_REQUEST_PAYER='requester',
                   CURL_CA_BUNDLE=certifi.where())
        #session = boto3.Session(aws_access_key_id=s3_cred['accessKeyId'], 
        #                aws_secret_access_key=s3_cred['secretAccessKey'],
        #                aws_session_token=s3_cred['sessionToken'],
        #                region_name='us-west-2')
        
        #rio_env = rio.Env(AWSSession(session),
        #          GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        #          GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
        #          GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'))
        #rio_env.__enter__()
        
    else:
        env = dict(GDAL_HTTP_MAX_RETRY='5',
                   GDAL_HTTP_RETRY_DELAY='2',
                   GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
                   AWS_NO_SIGN_REQUEST='YES',
                   GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
                   GDAL_SWATH_SIZE='200000000',
                   VSI_CURL_CACHE_SIZE='200000000',
                   GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                   GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'),
                   CURL_CA_BUNDLE=certifi.where())
    
    os.environ.update(env)
    
