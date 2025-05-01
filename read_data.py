import global_vars
import xarray as xr
import numpy as np
import dask

import utils


def read_atom(region_name):
    path_atom = global_vars.dir_atom

    #chunk_size_time = 640
    var_list_atom = ['N_fine_AMP', 'N_accum_AMP', 'N_coarse_AMP']
    ds_atom_0_in_per_cm3 = xr.open_mfdataset(path_atom+'*.nc',
                                             group='NAerosol',
                                             concat_dim = 'time',
                                             combine = 'nested',
                                             preprocess=lambda ds: ds[var_list_atom])


    ds_atom_lat_lon_alt = xr.open_mfdataset(path_atom+'*.nc',
                                            group='MMS',
                                            concat_dim = 'time',
                                            combine = 'nested',
                                            preprocess=lambda ds: ds[['G_LONG', 'G_LAT', 'G_ALT']])


    ds_atom_p = xr.open_mfdataset(path_atom+'*.nc',
                                  group='MMS',
                                  concat_dim = 'time',
                                  combine = 'nested',
                                  preprocess=lambda ds: ds[['P']])


    ds_atom_time = xr.open_mfdataset(path_atom+'*.nc',
                                     concat_dim = 'time',
                                     combine = 'nested')

    # Converting longitude from [-180°-x, 180°+x] to [0°,360°]
    ds_atom_lat_lon_alt['G_LONG'] = ds_atom_lat_lon_alt['G_LONG'] % 360

    # Converting particle number concentration from cm-3 to m-3 #
    ds_atom_0 = ds_atom_0_in_per_cm3 * (100 ** 3)

    # Building a new xr.Dataset from the three loaded xr.Datasets by combining the variables needed #
    ds_atom_0['time'] = ds_atom_time['time']
    ds_atom_p['time'] = ds_atom_time['time']
    ds_atom_lat_lon_alt['time'] = ds_atom_time['time']
    ds_atom = ds_atom_0.assign(lat=ds_atom_lat_lon_alt['G_LAT'], lon=ds_atom_lat_lon_alt['G_LONG'],
                               alt=ds_atom_lat_lon_alt['G_ALT'], p=ds_atom_p['P'])

    # Selecting the data measured at a height smaller than the hight limit and calculate 12h averages #
    height_lim = global_vars.height_limit
    lat_lon = utils.region_definition()[region_name]
    ds_atom_region = ds_atom.where((ds_atom['lat'].compute() > lat_lon['lat'][0]) &
                                   (ds_atom['lat'].compute() < lat_lon['lat'][1]) &
                                   (ds_atom['lon'].compute() > lat_lon['lon'][0]) &
                                   (ds_atom['lon'].compute() < lat_lon['lon'][1]),
                                drop='True')
    ds_atom_below_height_limit = ds_atom_region.where(ds_atom_region['alt'].compute() < height_lim,
                            drop='True')
    ds_atom_below_height_limit['time'] = ds_atom_below_height_limit.time.dt.floor(
        '12h')  # round time dimension, DDTHH:MM --> DDT00:00 or DDT12:00
    ds_atom_time_mean = ds_atom_below_height_limit.groupby('time').mean(
        skipna=True)  # calculate mean of each time chunk

    return ds_atom_time_mean



def read_model(ds_atom_time_mean):
    path_echam = global_vars.dir_echam
    exp = global_vars.experiment
    print(path_echam + '/' + exp + '*NUM_NUC_plev.nc')
    echam_nuc_meta = xr.open_mfdataset(path_echam + '/' + exp + '*NUM_NUC_plev.nc',
                                      concat_dim='time',
                                      combine='nested')

    # select relevant 12h-time chunks (half days)
    c_echam_meta_all_timesteps = echam_nuc_meta
    c_echam_meta_all_timesteps['time'] = c_echam_meta_all_timesteps.time.dt.ceil(
        '12h')  # use ceil() this time? DDTHH:MM --> DDT00:00 or DDT12:00

    # Only time steps covered by both ATom and ECHAM are used.
    sel_dates = ds_atom_time_mean.time.values[np.isin(ds_atom_time_mean.time.values, echam_nuc_meta.time.values)]

###############
    # "test mode"
    sel_dates = sel_dates[0:]
###############

    # coordinates of ATom flights
    sel_lats = ds_atom_time_mean.sel(time=sel_dates).lat
    sel_lons = ds_atom_time_mean.sel(time=sel_dates).lon
    sel_plevs = ds_atom_time_mean.sel(time=sel_dates).p * 100  # ATom pressure levels must be converted from hPa to Pa

    # load ECHAM number concentrations
    data_echam_num = global_vars.data_echam_number

    for mode_echam_i in data_echam_num.keys():
        print(path_echam + '/' + exp + data_echam_num[mode_echam_i]['file ending'])
        echam_0 = \
        xr.open_mfdataset(path_echam + '/' + exp + data_echam_num[mode_echam_i]['file ending'],
                          concat_dim='time',
                          combine='nested',
                          chunks={'plev': 30, 'lat': 30, 'lon': 30})[
            data_echam_num[mode_echam_i]['var. name']]
        echam_0['time'] = echam_0.time.dt.ceil('12h')
        echam_t = echam_0.sel(time=sel_dates)
        with dask.config.set(**{
            'array.slicing.split_large_chunks': True}):  # otherwise xarray produces a "PerformanceWarning: Slicing is producing a large chunk"
            echam_tp = echam_t.interp(plev=sel_plevs)
            echam_tpxy = echam_tp.interp(lat=sel_lats, lon=sel_lons)
        data_echam_num[mode_echam_i]['reduced dataset'] = echam_tpxy


    # load ECHAM median radii
    data_echam_rmed = global_vars.data_echam_rmedian

    for mode_echam_j in data_echam_rmed.keys():
        echam_rmed_0 = \
        xr.open_mfdataset(path_echam + '/' + exp + data_echam_rmed[mode_echam_j]['file ending'], concat_dim='time',
                          combine='nested', chunks={'plev': 30, 'lat': 30, 'lon': 30})[
            data_echam_rmed[mode_echam_j]['var. name']]
        echam_rmed_0['time'] = echam_rmed_0.time.dt.ceil('12h')
        echam_rmed_t = echam_rmed_0.sel(time=sel_dates)
        with dask.config.set(**{
            'array.slicing.split_large_chunks': True}):  # otherwise xarray produces a "PerformanceWarning: Slicing is producing a large chunk"
            echam_rmed_tp = echam_rmed_t.interp(plev=sel_plevs)
            echam_rmed_tpxy = echam_rmed_tp.interp(lat=sel_lats, lon=sel_lons)
        data_echam_rmed[mode_echam_j]['reduced dataset'] = echam_rmed_tpxy * (10 ** 6)  # convert into um


    return data_echam_num, data_echam_rmed, sel_dates


