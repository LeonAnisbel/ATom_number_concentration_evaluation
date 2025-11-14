import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import math
import global_vars
import utils

units = global_vars.units
figure_dir = global_vars.plot_dir


def read_nc_files_No_conc(ds_atom_time_mean, sel_dates, region_name, exp, mode_a):
    """
    Read netcdf files with ECHAM-HAM number concentration with modes mapped onto ATom modes
    :var ds_atom_time_mean: dataset with ATOM data with a 12 h mean
    :var sel_dates: time covered by both ATom and ECHAM
    :param region_name: region name
    :param exp: experiment name
    :param mode_a: mode name
    :return: 1-D number concentration from ATom and ECHAM-HAM
    """
    # read in netcdf-files
    nc_file_dir = global_vars.netcdf_file_dir
    c_fine_echam = xr.open_dataset(f'{nc_file_dir}/{exp}_c_fine_echam_{region_name}.nc')
    c_acc_echam = xr.open_dataset(f'{nc_file_dir}/{exp}_c_acc_echam_{region_name}.nc')
    c_coa_echam = xr.open_dataset(f'{nc_file_dir}/{exp}_c_coa_echam_{region_name}.nc')

    # %%
    # select the mode
    ds_of_mode = {'fine': {'atom': ds_atom_time_mean['N_fine_AMP'],
                           'echam': c_fine_echam['c_num']},
                  'accumulation': {'atom': ds_atom_time_mean['N_accum_AMP'],
                                   'echam': c_acc_echam['c_num']},
                  'coarse': {'atom': ds_atom_time_mean['N_coarse_AMP'],
                             'echam': c_coa_echam['c_num']}}

    c_atom = ds_of_mode[mode_a]['atom'].sel(time=sel_dates)
    c_echam_tpxy = ds_of_mode[mode_a]['echam']

    return c_atom, c_echam_tpxy

def plot_No_conc_ECHAM(data_echam_rmed, data_echam_num, region_name):
    """
    Plots number concentration of ECHAM without mapping the modes onto ATom modes
    :var data_echam_rmed: dataset with ECHAM-HAM data of median radius
    :var data_echam_num: dataset with ECHAM-HAM number concentration
    :param region_name: region name
    :return: None
    """
# inside certain latitude band (ECHAM data, time average of sum of all ECHAM modes)

    # define a latitude band
    lat_1 = 45
    lat_2 = 90
    parameters_echam = global_vars.params_echam
    n = []

    for mode_echam_k in parameters_echam.keys():
        # r_median_i for every time step
        r_median = data_echam_rmed[mode_echam_k]['reduced dataset'].where(
            data_echam_rmed[mode_echam_k]['reduced dataset'] > 0, other=np.nan)

        # sigma_i for every mode
        sigma_mode = parameters_echam[mode_echam_k]['sigma_i']

        r = np.logspace(-6, 1, 50)
        r_with_dim = xr.DataArray(data=r, dims=["radius"], coords={"r_values": ("radius", r)})
        n_k = (data_echam_num[mode_echam_k]['reduced dataset'] / (np.sqrt(2 * np.pi) * np.log(sigma_mode))) * np.exp(
            -((np.log(r_with_dim) - np.log(r_median)) ** 2) / (2 * np.log(sigma_mode) ** 2))
        n.append(n_k.where(n_k.lat > lat_1, other=np.nan).where(n_k.lat < lat_2, other=np.nan))

    n_sum = sum(n)

    fig, ax = plt.subplots(1, 1)
    ax.plot(n_sum.r_values, n_sum.mean(dim='time'))
    ax.set_xscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel('n')
    ax.set_title(f'{lat_1}° - {lat_2}°')
    plt.savefig(f'No_con_echam_{region_name}.png')
    plt.close()

    ##############################################
    # Calculate and plot the absolute difference #
    ##############################################
def plot_absolute_diff_map(ds_atom_time_mean, sel_dates, region_name):
    """
    Plots the absolute difference map of ATom values in contrast to model quantities of number concentration per mode
    :param ds_atom_time_mean: dataset with ATOM data with a 12 h mean
    :param sel_dates: time covered by both ATom and ECHAM
    :param region_name: region name
    :return: None
    """
    mode_atom = global_vars.mode_atom
    exp = global_vars.experiment
    c_atom, c_echam_tpxy = read_nc_files_No_conc(ds_atom_time_mean, sel_dates, region_name, exp, mode_atom)
    diff_abs_whole_time = c_echam_tpxy - c_atom

    # The time step with the maximum absolute difference between model and observations is calculated.
    print('Time step with the maximum absolute difference between model and observations: ' + str(
        abs(diff_abs_whole_time).idxmax(dim='time').values)[0:13])

    # %%
    t_start = sel_dates[0]
    t_end = sel_dates[-1]

    # in case one wants to define the time range manually:
    # t_start = '2016-07-30T00:00:00'
    # t_end = '2016-08-30T00:00:00'

    # %%
    colormap_range = math.ceil(
        abs(diff_abs_whole_time).max())  # define data range covered by colormap; make sure that it is symmetrical around 0

    diff_abs = diff_abs_whole_time.sel(time=slice(t_start, t_end))

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    map_plot = ax.scatter(diff_abs.lon, diff_abs.lat, c=diff_abs.values, cmap='RdBu', vmin=-colormap_range,
                          vmax=colormap_range, transform=ccrs.PlateCarree())
    ax.coastlines()
    cbar = fig.colorbar(map_plot)
    ax.set_title(
        f'$c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ECHAM}} - c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}}$ in m$^{{-3}}$\n{str(t_start)[0:10]} - {str(t_end)[0:10]}\n{exp}')

    # add grid lines with labels
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False


    plt.savefig(f'{figure_dir}/diff_abs_{str(t_start)[0:10]}-{str(t_end)[0:10]}_{exp}_{mode_atom}_{region_name}.pdf',
                bbox_inches='tight')
    plt.close()

def plot_relat_diff_map(ds_atom_time_mean, sel_dates, region_name):
    """
    Plots the relative difference map of ATom values in contrast to model quantities of number concentration per mode
    :param ds_atom_time_mean: dataset with ATOM data with a 12 h mean
    :param sel_dates: time covered by both ATom and ECHAM
    :param region_name: region name
    :return: None
    """
    mode_atom = global_vars.mode_atom
    exp = global_vars.experiment
    c_atom, c_echam_tpxy = read_nc_files_No_conc(ds_atom_time_mean, sel_dates, region_name, exp, mode_atom)
    diff_rel_whole_time = (c_echam_tpxy - c_atom) / c_atom

    # The time step with the maximum relative difference between model and observations is calculated.
    print('Time step with the maximum relative difference between model and observations: ' + str(
        abs(diff_rel_whole_time).idxmax(dim='time').values)[0:10])

    # %%
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
    # colormap_range = math.ceil(abs(diff_rel).max(skipna=True))

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    map_plot = ax.scatter(diff_rel.lon, diff_rel.lat, c=diff_rel.values, cmap='RdBu', vmin=-colormap_range,
                          vmax=colormap_range, transform=ccrs.PlateCarree())
    ax.coastlines()
    cbar = fig.colorbar(map_plot)
    exp = global_vars.experiment
    ax.set_title(
        f'$(c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ECHAM}} - c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}})/c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}}$\n{str(t_start)[0:10]}-{str(t_end)[0:10]}\n{exp}')

    # add grid lines with labels
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    figure_dir = global_vars.plot_dir
    plt.savefig(f'{figure_dir}/diff_rel_{str(t_start)[0:10]}-{str(t_end)[0:10]}_{exp}_{mode_atom}_new_{region_name}.pdf',
                bbox_inches='tight')
    plt.close()

    statistical_quantities = utils.statistics(c_atom, c_echam_tpxy, sel_dates)
    with open(f'{figure_dir}/statistical_quantities_whole_time_{exp}_{mode_atom}_{region_name}.txt', 'w') as f:
        for statistical_quantity in statistical_quantities:
            f.write(f'{statistical_quantity[0]} = {statistical_quantity[1]:.2f} {statistical_quantity[2]}\n')

    # %%
    # replace ':.2f' with ':.2e' for scientific notation
    # for statistical_quantity in statistical_quantities:
    # %%
    ################
    # Scatter plot #
    ################
def scatter_plot(ds_atom_time_mean, sel_dates, region_name):
    """
    Plots the data of ATom values in contrast to model quantities of number concentration per mode
    :param ds_atom_time_mean: dataset with ATOM data with a 12 h mean
    :param sel_dates: time covered by both ATom and ECHAM
    :param region_name: region name
    :return: None
    """
    mode_atom = global_vars.mode_atom
    exp = global_vars.experiment
    c_atom, c_echam_tpxy = read_nc_files_No_conc(ds_atom_time_mean, sel_dates, region_name, exp, mode_atom)

    print(f'Scatter plot')
    t_start = sel_dates[0]
    t_end = sel_dates[-1]

    # filter out nan values as np.polynomial.Polynomial.fit() cannot deal with NaN's
    mask_nan_atom = np.isnan(c_atom.sel(time=slice(t_start, t_end)))
    mask_nan_echam = np.isnan(c_echam_tpxy.sel(time=slice(t_start, t_end)))

    c_atom_wo_nan = c_atom.sel(time=slice(t_start, t_end)).where(~mask_nan_atom, drop=True).where(~mask_nan_echam,
                                                                                                  drop=True)
    c_echam_tpxy_wo_nan = c_echam_tpxy.sel(time=slice(t_start, t_end)).where(~mask_nan_atom, drop=True).where(
        ~mask_nan_echam, drop=True)

    # plot
    fig = plt.figure()
    ax_scatter = plt.axes()
    ax_scatter.plot(c_atom_wo_nan, c_echam_tpxy_wo_nan, marker='.', markeredgecolor='none', markersize=4, ls='', zorder=2)
    # ax_scatter.set_aspect(aspect=1)

    ax_scatter.set_xlabel(f'$c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ATom}}$ ({units})')
    ax_scatter.set_ylabel(f'$c_\mathrm{{{mode_atom}}}$$_\mathrm{{,ECHAM}}$ ({units})')

    linreg_coeffs = np.polynomial.Polynomial.fit(c_atom_wo_nan, c_echam_tpxy_wo_nan,
                                                 deg=1).convert().coef  # perform linear regression, get intercept and slope
    x = np.linspace(c_atom_wo_nan.min(), c_atom_wo_nan.max(), num=2)  # define x-values of regression line
    y = linreg_coeffs[0] + linreg_coeffs[1] * x  # define y-values of regression line
    ax_scatter.plot(x, y, label=f'linear regression\ny = {linreg_coeffs[1]:.2f} x + {linreg_coeffs[0]:.2f}',
                    zorder=3)  # draw regression line
    x_eq = np.array(
        [max(c_echam_tpxy_wo_nan.min(), c_atom_wo_nan.min()), min(c_echam_tpxy_wo_nan.max(), c_atom_wo_nan.max())])
    ax_scatter.plot(x_eq, x_eq, label='y = x', zorder=1)
    ax_scatter.set_title(f'{str(t_start)[0:10]}-{str(t_end)[0:10]}\n{exp}')
    ax_scatter.legend()
    ax_scatter.grid(True)

    statistical_quantities = utils.statistics(c_atom, c_echam_tpxy, sel_dates)

    for i in range(len(statistical_quantities)):
        plt.gcf().text(0.11, -(i + 3) * 0.04,
                       f'{statistical_quantities[i][0]} = {statistical_quantities[i][1]:.2e} {statistical_quantities[i][2]}')

    plt.savefig(f'{figure_dir}/scatter_{str(t_start)[0:10]}-{str(t_end)[0:10]}_{exp}_{mode_atom}_{region_name}.pdf',
                bbox_inches='tight')
    plt.show()
    plt.close()

    # %%

