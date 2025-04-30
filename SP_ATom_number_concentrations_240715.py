#%%
import cartopy.crs as ccrs
import dask
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import time
import xarray as xr

# new netcdf-files will be written; specify where they shall be written
# # also specify the directory for the created figures
nc_file_dir = '/vols/fs1/work/paul/scripts'
figure_dir = '/vols/fs1/work/paul/figures/ATom_vs_ECHAM/particle_number_concentrations'

#%%
#####################
# Loading ATom data #
#####################

# The variables are selected explicitely, because they are not the same throughout all files and thus automatically merging them doesn't work. Also, this way, less memory is needed.

path_atom = '/vols/fs1/work/leon/ATom_vs_ECHAM/ATom-merge_ncfiles/MER-SAGA'

#chunk_size_time = 640
var_list_atom = ['N_fine_AMP', 'N_accum_AMP', 'N_coarse_AMP']
#ds_atom_0_in_per_cm3 = xr.open_mfdataset(path_atom+'*.nc', group='NAerosol', concat_dim = 'time', combine = 'nested', chunks={'time': chunk_size_time}, preprocess=lambda ds: ds[var_list_atom])
ds_atom_0_in_per_cm3 = xr.open_mfdataset(path_atom+'*.nc', group='NAerosol', concat_dim = 'time', combine = 'nested', preprocess=lambda ds: ds[var_list_atom])


# ds_atom_lat_lon_alt = xr.open_mfdataset(path_atom+'*.nc', group='AMS', concat_dim = 'time', combine = 'nested', preprocess=lambda ds: ds[['LON_AMS', 'LAT_AMS', 'ALT_AMS']])
#ds_atom_lat_lon_alt = xr.open_mfdataset(path_atom+'*.nc', group='MMS', concat_dim = 'time', combine = 'nested', chunks={'time': chunk_size_time}, preprocess=lambda ds: ds[['G_LONG', 'G_LAT', 'G_ALT']])
ds_atom_lat_lon_alt = xr.open_mfdataset(path_atom+'*.nc', group='MMS', concat_dim = 'time', combine = 'nested', preprocess=lambda ds: ds[['G_LONG', 'G_LAT', 'G_ALT']])


#ds_atom_p = xr.open_mfdataset(path_atom+'*.nc', group='MMS', concat_dim = 'time', combine = 'nested', chunks={'time': chunk_size_time}, preprocess=lambda ds: ds[['P']])
ds_atom_p = xr.open_mfdataset(path_atom+'*.nc', group='MMS', concat_dim = 'time', combine = 'nested', preprocess=lambda ds: ds[['P']])


#ds_atom_time = xr.open_mfdataset(path_atom+'*.nc', concat_dim = 'time', combine = 'nested', chunks={'time': chunk_size_time})
ds_atom_time = xr.open_mfdataset(path_atom+'*.nc', concat_dim = 'time', combine = 'nested')


# [N_atom] = cm^-3


#%%
############################################################
# Converting longitude from [-180°-x, 180°+x] to [0°,360°] #
############################################################

# The longitude is constrained to [-180°, 180°] and converted to the interval [0°, 360°], because the ATom-flights cross the -180°/180°-longitude, but do not cross the 0°/360°-longitude; this way, there will
# l be no problems when averaging the longitude values. For other flights, one will have to check whether this is still the case.

ds_atom_lat_lon_alt['G_LONG'] = ds_atom_lat_lon_alt['G_LONG'] % 360



#%%
#############################################################
# Converting particle number concentration from cm-3 to m-3 #
#############################################################

ds_atom_0 = ds_atom_0_in_per_cm3*(100**3)

# [N_atom] = m^-3

#%%
#################################################################################################
# Building a new xr.Dataset from the three loaded xr.Datasets by combining the variables needed #
#################################################################################################

ds_atom_0['time'] = ds_atom_time['time']
ds_atom_p['time'] = ds_atom_time['time']
ds_atom_lat_lon_alt['time'] = ds_atom_time['time']
ds_atom = ds_atom_0.assign(lat=ds_atom_lat_lon_alt['G_LAT'], lon=ds_atom_lat_lon_alt['G_LONG'], alt=ds_atom_lat_lon_alt['G_ALT'], p=ds_atom_p['P'])



#%%
###################################################################################################
# Selecting the data measured at a height smaller than the hight limit and calculate 12h averages #
###################################################################################################

height_limit = 1000 # height limit in m

mask_below_height_limit = ds_atom.alt < height_limit
ds_atom_below_height_limit_0 = ds_atom.where(mask_below_height_limit, drop=True)
ds_atom_below_height_limit = ds_atom.where(mask_below_height_limit, drop=True) # remove all time steps where altitude is >= height limit
ds_atom_below_height_limit['time'] = ds_atom_below_height_limit.time.dt.floor('12h') # round time dimension, DDTHH:MM --> DDT00:00 or DDT12:00 
ds_atom_time_mean = ds_atom_below_height_limit.groupby('time').mean(skipna=True) # calculate mean of each time chunk

# If measured values of a certain variable in ds_atom_time_mean are == nan, it is because at this 12h-period all measured values were == nan.



# %%
##############################################
# Select relevant time steps and coordinates #
##############################################

# Consider using the 'test mode' first.

run = 'echam_base'
# run = 'ac3_arctic'

path_echam = f'/vols/fs1/work/leon/ATom_vs_ECHAM/ECHAM_HAM_output/{run}/NUM_radius'
echam_nuc_meta = xr.open_mfdataset(path_echam+'/'+run+'*NUM_NUC_plev.nc', concat_dim = 'time', combine = 'nested')

# select relevant 12h-time chunks (half days)
c_echam_meta_all_timesteps = echam_nuc_meta
c_echam_meta_all_timesteps['time'] = c_echam_meta_all_timesteps.time.dt.ceil('12h') # use ceil() this time? DDTHH:MM --> DDT00:00 or DDT12:00

# Only time steps covered by both ATom and ECHAM are used.
sel_dates = ds_atom_time_mean.time.values[np.isin(ds_atom_time_mean.time.values, echam_nuc_meta.time.values)]

# "test mode"
sel_dates = sel_dates[0:5]

#%%
# coordinates of ATom flights
sel_lats = ds_atom_time_mean.sel(time=sel_dates).lat
sel_lons = ds_atom_time_mean.sel(time=sel_dates).lon
sel_plevs = ds_atom_time_mean.sel(time=sel_dates).p * 100 # ATom pressure levels must be converted from hPa to Pa

# ECHAM pressure levels are in Pa

#%%
#################################
# Loading ECHAM-HAM output data #
#################################

# load ECHAM number concentrations
data_echam_num = {
    'nucleation': {'file ending': '*NUM_NUC_plev.nc', 'var. name': 'NUM_NUC'},
    'aitken_i': {'file ending': '*NUM_KI_plev.nc', 'var. name': 'NUM_KI'},
    'aitken_s': {'file ending': '*NUM_KS_plev.nc', 'var. name': 'NUM_KS'},
    'accumulation_i': {'file ending': '*NUM_AI_plev.nc', 'var. name': 'NUM_AI'},
    'accumulation_s': {'file ending': '*NUM_AS_plev.nc', 'var. name': 'NUM_AS'},
    'coarse_i': {'file ending': '*NUM_CI_plev.nc', 'var. name': 'NUM_CI'},
    'coarse_s': {'file ending': '*NUM_CS_plev.nc', 'var. name': 'NUM_CS'}
    }

for mode_echam_i in data_echam_num.keys():
    echam_0 = xr.open_mfdataset(path_echam+'/'+run+data_echam_num[mode_echam_i]['file ending'], concat_dim = 'time', combine = 'nested', chunks={'plev': 30, 'lat': 30, 'lon': 30})[data_echam_num[mode_echam_i]['var. name']]
    echam_0['time'] = echam_0.time.dt.ceil('12h')
    echam_t = echam_0.sel(time=sel_dates)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}): # otherwise xarray produces a "PerformanceWarning: Slicing is producing a large chunk"
        echam_tp = echam_t.interp(plev=sel_plevs)
        echam_tpxy = echam_tp.interp(lat=sel_lats, lon=sel_lons)
    data_echam_num[mode_echam_i]['reduced dataset'] = echam_tpxy



#%%
# load ECHAM median radii
data_echam_rmed = {
    'nucleation': {'file ending': '*m_radius_NUC_plev.nc', 'var. name': 'rwet_NUC'},
    'aitken_i': {'file ending': '*m_radius_KI_plev.nc', 'var. name': 'rdry_KI'},
    'aitken_s': {'file ending': '*m_radius_KS_plev.nc', 'var. name': 'rwet_KS'},
    'accumulation_i': {'file ending': '*m_radius_AI_plev.nc', 'var. name': 'rdry_AI'},
    'accumulation_s': {'file ending': '*m_radius_AS_plev.nc', 'var. name': 'rwet_AS'},
    'coarse_i': {'file ending': '*m_radius_CI_plev.nc', 'var. name': 'rdry_CI'},
    'coarse_s': {'file ending': '*m_radius_CS_plev.nc', 'var. name': 'rwet_CS'}
    }

for mode_echam_j in data_echam_rmed.keys():
    echam_rmed_0 = xr.open_mfdataset(path_echam+'/'+run+data_echam_rmed[mode_echam_j]['file ending'], concat_dim = 'time', combine = 'nested', chunks={'plev': 30, 'lat': 30, 'lon': 30})[data_echam_rmed[mode_echam_j]['var. name']]
    echam_rmed_0['time'] = echam_rmed_0.time.dt.ceil('12h')
    echam_rmed_t = echam_rmed_0.sel(time=sel_dates)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}): # otherwise xarray produces a "PerformanceWarning: Slicing is producing a large chunk"
        echam_rmed_tp = echam_rmed_t.interp(plev=sel_plevs)
        echam_rmed_tpxy = echam_rmed_tp.interp(lat=sel_lats, lon=sel_lons)
    data_echam_rmed[mode_echam_j]['reduced dataset'] = echam_rmed_tpxy*(10**6) # convert into um


#%%
########################################################################################################
# calculate number of particles in ECHAM that go into ATom fine, accumulation and coarse, respectively #
########################################################################################################

parameters_echam = {
    'nucleation': {'sigma_i': 1.59},
    'aitken_i': {'sigma_i': 1.59},
    'aitken_s': {'sigma_i': 1.59},
    'accumulation_i': {'sigma_i': 1.59},
    'accumulation_s': {'sigma_i': 1.59},
    'coarse_i': {'sigma_i': 2},
    'coarse_s': {'sigma_i': 2}
    }



#%%
# everything is radius, in um

parts_fine_echam = []
parts_acc_echam = []
parts_coa_echam = []

for mode_echam_k in parameters_echam.keys():

    # r_median_i for every time step
    r_median = data_echam_rmed[mode_echam_k]['reduced dataset'].where(data_echam_rmed[mode_echam_k]['reduced dataset'] > 0, other=np.nan)

    # sigma_i for every mode
    sigma_mode = parameters_echam[mode_echam_k]['sigma_i']

    ##############################
    # calculate parts of total N #
    ##############################

    # part of integral that should be in ATom fine mode
    # ATom mode limits
    fine_atom_lower = 0.00135
    fine_atom_upper = 0.25

    # ATom mode limits taking into account subsitution
    alpha_atom_fine = (np.log(fine_atom_lower)-np.log(r_median))/np.log(sigma_mode)
    beta_atom_fine = (np.log(fine_atom_upper)-np.log(r_median))/np.log(sigma_mode)

    # number of particles in ECHAM inside ATom mode limits (with N = 1)
    N_atom_fine = (norm.cdf(beta_atom_fine)-norm.cdf(alpha_atom_fine))

    # fraction of N_tot that should be counted as ATom fine mode
    fraction_fine = (r_median/r_median) * N_atom_fine # = N_atom_fine/1; (r_median/r_median) is just to keep the xarray-dimensions

    # number of particles that should be counted as ATom fine mode
    part_fine_echam_from_mode = fraction_fine*data_echam_num[mode_echam_k]['reduced dataset']
    parts_fine_echam.append(part_fine_echam_from_mode)

    # part of integral that should be in ATom accumulation mode
    # ATom mode limits
    acc_atom_lower = 0.03
    acc_atom_upper = 0.25

    # ATom mode limits taking into account subsitution
    alpha_atom_acc = (np.log(acc_atom_lower)-np.log(r_median))/np.log(sigma_mode)
    beta_atom_acc = (np.log(acc_atom_upper)-np.log(r_median))/np.log(sigma_mode)

    # number of particles in ECHAM inside ATom mode limits (with N = 1)
    N_atom_acc = (norm.cdf(beta_atom_acc)-norm.cdf(alpha_atom_acc))

    # fraction of N_tot that should be counted as ATom accumulation mode
    fraction_acc = (r_median/r_median) * N_atom_acc # = N_atom_acc/1; (r_median/r_median) is just to keep the xarray-dimensions

    # number of particles that should be counted as ATom accumulation mode
    part_acc_echam_from_mode = fraction_acc*data_echam_num[mode_echam_k]['reduced dataset']
    parts_acc_echam.append(part_acc_echam_from_mode)



    # part of integral that should be in ATom coarse mode
    # ATom mode limits
    coa_atom_lower = 0.25
    coa_atom_upper = 2.4

    # ATom mode limits taking into account subsitution
    alpha_atom_coa = (np.log(coa_atom_lower)-np.log(r_median))/np.log(sigma_mode)
    beta_atom_coa = (np.log(coa_atom_upper)-np.log(r_median))/np.log(sigma_mode)

    # number of particles in ECHAM inside ATom mode limits (with N = 1)
    N_atom_coa = (norm.cdf(beta_atom_coa)-norm.cdf(alpha_atom_coa))

    # fraction of N_tot that should be counted as ATom coarse mode
    fraction_coa = (r_median/r_median)* N_atom_coa # = N_atom_coa/1; (r_median/r_median) is just to keep the xarray-dimensions

    # number of particles that should be counted as ATom coarse mode
    part_coa_echam_from_mode = fraction_coa*data_echam_num[mode_echam_k]['reduced dataset']
    parts_coa_echam.append(part_coa_echam_from_mode)




    ####################################################################
    # visualize example n(r) with respective integral ATom mode limits #
    ####################################################################

    # the integral is calculated a) purely numerically and b) with help of substitution and standard normal cumulative distribution function
    # by plotting, one can check whether they match

    # randomly pick one example distribution
    r_med_example = float(r_median.isel(time=0))
    r_example = np.linspace(0.001, max(r_med_example*5+0.001,coa_atom_upper+1.01), 101)

    # lognormal distribution function
    log_normal_pdf_example = lambda r:  (1/(np.sqrt(2*np.pi)*np.log(sigma_mode)))*np.exp(-((np.log(r)-np.log(r_med_example))**2)/(2*np.log(sigma_mode)**2))
    n_example = log_normal_pdf_example(r_example)

    # a) purely numerical calculation
    N_example_num = []
    for r_i in r_example: # numerical calculation of function's integral
        N_example_num_i = quad(log_normal_pdf_example, 0, r_i)[0]
        N_example_num.append(N_example_num_i)

    # b) calculation with substitution and standard normal cumulative distribution function
    r_example_subs = (np.log(r_example)-np.log(r_med_example)-np.log(sigma_mode)**2)/np.log(sigma_mode)
    N_example_ncdf = np.exp(np.log(r_med_example)+(np.log(sigma_mode)**2)/2) * (norm.cdf(r_example_subs))

    fig, ax = plt.subplots(1, 1)
    ax.plot(r_example, n_example, color = 'cornflowerblue') # lognormal distribution
    ax.plot(r_example, N_example_num, color = 'orange') # integral of lognormal distribution, calculated purely numerically
    ax.plot(r_example, N_example_ncdf, color = 'green') # integral of lognormal distribution, calculated with standard normal cumulative distribution function

    ax.set_xlim(right=2.7)

    ax.vlines(fine_atom_lower, 0, 0.9, color = 'blueviolet')
    ax.text(fine_atom_lower, 0, 'fine_atom_lower', rotation = 'vertical')

    ax.vlines(acc_atom_lower, 0, 0.9, color = 'mediumvioletred')
    ax.text(acc_atom_lower, 0.5, 'acc_atom_lower', rotation = 'vertical')

    ax.vlines(acc_atom_upper, 0, 0.9, color = 'firebrick')
    ax.text(acc_atom_upper, 0, 'fine/acc to coa', rotation = 'vertical')

    ax.vlines(coa_atom_upper, 0, 0.9, color = 'chocolate')
    ax.text(coa_atom_upper, 0, 'coa_atom_upper', rotation = 'vertical')

    ax.set_title('ECHAM number distribution for example r_median_i,\nintegral of number distribution\nand ATom mode limits')

    plt.gcf().text(0.11, -(0+2)*0.04, f'part of "fine" in ECHAM from this mode = {float(fraction_fine.isel(time=0))*100} %')
    plt.gcf().text(0.11, -(1+2)*0.04, f'part of "accumulation" in ECHAM from this mode = {float(fraction_acc.isel(time=0))*100} %')
    plt.gcf().text(0.11, -(2+2)*0.04, f'part of "coarse" in ECHAM from this mode = {float(fraction_coa.isel(time=0))*100} %')

    plt.show()
    plt.close()
    # If the orange line is covered by the green line, they match!
#%%
# sum up all contributions to ATom modes and save it in new netcdf-files; this seems to be faster
c_fine_echam_0 = sum(parts_fine_echam)
c_fine_echam_ds = c_fine_echam_0.to_dataset(name='c_num')
c_fine_echam_ds.to_netcdf(f'{nc_file_dir}/{run}_c_fine_echam.nc')

c_acc_echam_0 = sum(parts_acc_echam)
c_acc_echam_ds = c_acc_echam_0.to_dataset(name='c_num')
c_acc_echam_ds.to_netcdf(f'{nc_file_dir}/{run}_c_acc_echam.nc')

c_coa_echam_0 = sum(parts_coa_echam)
c_coa_echam_ds = c_coa_echam_0.to_dataset(name='c_num')
c_coa_echam_ds.to_netcdf(f'{nc_file_dir}/{run}_c_coa_echam.nc')





#%%
#####################
# Plot n(r) (ECHAM) #
#####################

# inside certain latitude band (ECHAM data, time average of sum of all ECHAM modes)

# define a latitude band
lat_1 = 45
lat_2 = 90

n = []

for mode_echam_k in parameters_echam.keys():

    # r_median_i for every time step
    r_median = data_echam_rmed[mode_echam_k]['reduced dataset'].where(data_echam_rmed[mode_echam_k]['reduced dataset'] > 0, other=np.nan)

    # sigma_i for every mode
    sigma_mode = parameters_echam[mode_echam_k]['sigma_i']

    r = np.logspace(-6, 1, 50)
    r_with_dim = xr.DataArray(data=r, dims=["radius"], coords={"r_values": ("radius", r)})
    n_k = (data_echam_num[mode_echam_k]['reduced dataset']/(np.sqrt(2*np.pi)*np.log(sigma_mode)))*np.exp(-((np.log(r_with_dim)-np.log(r_median))**2)/(2*np.log(sigma_mode)**2))
    n.append(n_k.where(n_k.lat > lat_1, other=np.nan).where(n_k.lat < lat_2, other=np.nan))

n_sum = sum(n)

fig, ax = plt.subplots(1, 1)
ax.plot(n_sum.r_values, n_sum.mean(dim='time'))
ax.set_xscale('log')
ax.set_xlabel('r')
ax.set_ylabel('n')
ax.set_title(f'{lat_1}° - {lat_2}°')
plt.show()
plt.close()

#%%
# read in netcdf-files
c_fine_echam = xr.open_dataset(f'{nc_file_dir}/{run}_c_fine_echam.nc')
c_acc_echam = xr.open_dataset(f'{nc_file_dir}/{run}_c_acc_echam.nc')
c_coa_echam = xr.open_dataset(f'{nc_file_dir}/{run}_c_coa_echam.nc')



#%%
# select the mode
mode_atom = 'coarse'
ds_of_mode = {'fine': {'atom': ds_atom_time_mean['N_fine_AMP'], 'echam': c_fine_echam['c_num']},
                 'accumulation': {'atom': ds_atom_time_mean['N_accum_AMP'], 'echam': c_acc_echam['c_num']},
                 'coarse': {'atom': ds_atom_time_mean['N_coarse_AMP'], 'echam': c_coa_echam['c_num']}}

c_atom = ds_of_mode[mode_atom]['atom'].sel(time=sel_dates)
c_echam_tpxy = ds_of_mode[mode_atom]['echam']





#%%
#############
# Map plots #
#############

##############################################
# Calculate and plot the absolute difference #
##############################################

diff_abs_whole_time = c_echam_tpxy - c_atom

# The time step with the maximum absolute difference between model and observations is calculated.
print('Time step with the maximum absolute difference between model and observations: ' + str(abs(diff_abs_whole_time).idxmax(dim='time').values)[0:13])



#%%
t_start = sel_dates[0]
t_end = sel_dates[-1]

# in case one wants to define the time range manually:
# t_start = '2016-07-30T00:00:00'
# t_end = '2016-08-30T00:00:00'

#%%
colormap_range = math.ceil(abs(diff_abs_whole_time).max()) # define data range covered by colormap; make sure that it is symmetrical around 0

diff_abs = diff_abs_whole_time.sel(time=slice(t_start,t_end))

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
map_plot = ax.scatter(diff_abs.lon, diff_abs.lat, c=diff_abs.values, cmap='RdBu', vmin=-colormap_range, vmax=colormap_range, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(map_plot)
ax.set_title(f'$c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ECHAM}} - c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}}$ in $m^{{-3}}$\n{str(t_start)[0:10]} - {str(t_end)[0:10]}\n{run}')

# add grid lines with labels
gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

plt.savefig(f'{figure_dir}/diff_abs_{str(t_start)[0:10]}-{str(t_end)[0:10]}_{run}_{mode_atom}.pdf', bbox_inches='tight')
plt.show()
plt.close()


#%%
##############################################
# Calculate and plot the relative difference #
##############################################

diff_rel_whole_time = (c_echam_tpxy - c_atom)/c_atom

# The time step with the maximum relative difference between model and observations is calculated.
print('Time step with the maximum relative difference between model and observations: ' + str(abs(diff_rel_whole_time).idxmax(dim='time').values)[0:10])



#%%
t_start = sel_dates[0]
t_end = sel_dates[-1]

# in case one wants to define the time range manually:
# t_start = '2016-07-30T00:00:00'
# t_end = '2016-12-30T00:00:00'

diff_rel = diff_rel_whole_time.sel(time=slice(t_start, t_end))

# in order to use the same colormap range to be the same throughout all plots, use the following line
colormap_range = math.ceil(abs(diff_rel).max(skipna=True))

mask_inf = diff_rel == np.inf
diff_rel = diff_rel.where(~mask_inf, other=np.nan)

# in order to use a colormap range optimized for the individual plot, use the following line
#colormap_range = math.ceil(abs(diff_rel).max(skipna=True))

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
map_plot = ax.scatter(diff_rel.lon, diff_rel.lat, c=diff_rel.values, cmap='RdBu', vmin=-colormap_range, vmax=colormap_range, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(map_plot)
ax.set_title(f'$(c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ECHAM}} - c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}})/c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}}$\n{str(t_start)[0:10]}-{str(t_end)[0:10]}\n{run}')

# add grid lines with labels
gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

plt.savefig(f'{figure_dir}/diff_rel_{str(t_start)[0:10]}-{str(t_end)[0:10]}_{run}_{mode_atom}_new.pdf', bbox_inches='tight')
plt.show()
plt.close()

#%%
################################################
# Calculating statistics for whole time period #
################################################
print('Start calculating statistics')

num_timesteps = len(sel_dates)

# model
std_model = float(np.nanstd(c_echam_tpxy, ddof=1)) # model data's standard deviation at station location
mean_model = float(np.nanmean(c_echam_tpxy)) # model data's mean at station location

# observations
std_obs = float(np.nanstd(c_atom, ddof=1)) # standard deviation of observed values
mean_obs = float(np.nanmean(c_atom)) # mean of observed values over time

# model-observations
RMSE = float(math.sqrt(np.square(np.subtract(c_atom, c_echam_tpxy)).mean(skipna=True))) # root mean square error btw. model and observations at station location
mean_bias = np.nanmean(np.subtract(c_echam_tpxy, c_atom))  # mean bias btw. model and observation at station location
normalized_mean_bias = np.nansum(np.subtract(c_echam_tpxy, c_atom))/np.nansum(c_atom)  # normalized mean biases btw. model and observations, at station locations
pearsons_coeff = np.corrcoef(c_atom, c_echam_tpxy)[0,1]  # Pearson's correlation coefficient
ioa = 1 - (np.nansum(np.square(c_atom-c_echam_tpxy)))/(np.nansum(np.square(np.abs(c_echam_tpxy-np.nanmean(c_atom))+np.abs(c_atom-np.nanmean(c_atom))))) # index of agreement
mean_fractional_bias = np.nanmean(2*(c_echam_tpxy-c_atom)/(c_echam_tpxy+c_atom))  # mean fractional bias
mean_fractional_error = np.nanmean(2*np.abs(c_echam_tpxy-c_atom)/(c_echam_tpxy+c_atom))  # mean fractional errors
m_t_wrt_o_t = c_echam_tpxy.values/c_atom.values  # model values with respect to obs. values
fac2 = np.count_nonzero(np.logical_and(0.5 <= m_t_wrt_o_t, m_t_wrt_o_t <= 2))/num_timesteps   # fraction of complete data pairs where model is within a factor of 2 of observation
fac10 = np.count_nonzero(np.logical_and(0.1 <= m_t_wrt_o_t, m_t_wrt_o_t <= 10))/num_timesteps  # fraction of complete data pairs where model is within a factor of 10 of observation

#%%
#units = c_echam_tpxy.attrs['units']
units = '$m^{{-3}}$'

statistical_quantities = [
        ['mean standard deviation (of model, at station location)', std_model, units],
        ['model mean (at station location)', mean_model, units],
        ['standard deviation of observations', std_obs, units],
        ['mean of observations', mean_obs, units],
        ['mean RMSE (btw. model and observations, at station location)', RMSE, units],
        ['mean bias (btw. model and observations)', mean_bias, units],
        ['normalized mean bias (btw. model and observations, at station location)', normalized_mean_bias, units],
        ["Pearson's correlation coefficient (btw. model and observations, at station location)", pearsons_coeff, ''],
        ['index of agreement (btw. model and observations, at station location)', ioa, ''],
        ['mean fractional bias (btw. model and observations, at station location)', mean_fractional_bias, ''],
        ['mean fractional error (btw. model and observations, at station location)', mean_fractional_error, ''],
        ['fraction of complete data pairs where model (at station location) is within a factor of 2 of obs.', fac2, ''],
        ['fraction of complete data pairs where model (at station location) is within a factor of 10 of obs.', fac10, '']
    ]

with open(f'{figure_dir}/statistical_quantities_whole_time_{run}_{mode_atom}.txt', 'w') as f:
    for statistical_quantity in statistical_quantities:
        f.write(f'{statistical_quantity[0]} = {statistical_quantity[1]:.2f} {statistical_quantity[2]}\n')

#%%
# replace ':.2f' with ':.2e' for scientific notation
#for statistical_quantity in statistical_quantities:
#%%
################
# Scatter plot #
################
print(f'Scatter plot')

# filter out nan values as np.polynomial.Polynomial.fit() cannot deal with NaN's
mask_nan_atom = np.isnan(c_atom.sel(time=slice(t_start,t_end)))
mask_nan_echam = np.isnan(c_echam_tpxy.sel(time=slice(t_start,t_end)))

c_atom_wo_nan = c_atom.sel(time=slice(t_start,t_end)).where(~mask_nan_atom, drop=True).where(~mask_nan_echam, drop=True)
c_echam_tpxy_wo_nan = c_echam_tpxy.sel(time=slice(t_start,t_end)).where(~mask_nan_atom, drop=True).where(~mask_nan_echam, drop=True)

# plot
fig = plt.figure()
ax_scatter = plt.axes()
ax_scatter.plot(c_atom_wo_nan, c_echam_tpxy_wo_nan, marker='.', markeredgecolor='none', markersize=4, ls='', zorder=2)
#ax_scatter.set_aspect(aspect=1)

ax_scatter.set_xlabel(f'$c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}}$ ({units})')
ax_scatter.set_ylabel(f'$c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ECHAM}}$ ({units})')

linreg_coeffs = np.polynomial.Polynomial.fit(c_atom_wo_nan, c_echam_tpxy_wo_nan, deg=1).convert().coef # perform linear regression, get intercept and slope
x = np.linspace(c_atom_wo_nan.min(), c_atom_wo_nan.max(), num=2)  # define x-values of regression line
y = linreg_coeffs[0]+linreg_coeffs[1]*x  # define y-values of regression line
ax_scatter.plot(x, y, label=f'linear regression\ny = {linreg_coeffs[1]:.2f} x + {linreg_coeffs[0]:.2f}', zorder=3)  # draw regression line
x_eq = np.array([max(c_echam_tpxy_wo_nan.min(), c_atom_wo_nan.min()), min(c_echam_tpxy_wo_nan.max(), c_atom_wo_nan.max())])
ax_scatter.plot(x_eq, x_eq, label='y = x', zorder=1)
ax_scatter.set_title(f'{str(t_start)[0:10]}-{str(t_end)[0:10]}\n{run}')
ax_scatter.legend()
ax_scatter.grid(True)

for i in range(len(statistical_quantities)):
    plt.gcf().text(0.11, -(i+3)*0.04, f'{statistical_quantities[i][0]} = {statistical_quantities[i][1]:.2e} {statistical_quantities[i][2]}')

plt.savefig(f'{figure_dir}/scatter_{str(t_start)[0:10]}-{str(t_end)[0:10]}_{run}_{mode_atom}.pdf', bbox_inches='tight')
plt.show()
plt.close()

# %%
                                                            
