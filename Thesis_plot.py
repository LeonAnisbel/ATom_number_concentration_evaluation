import pickle

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import math

from docutils.nodes import title

import global_vars
import utils
from global_vars import experiment, mode_atom
from plots import read_nc_files_No_conc

font = 12
experiments = global_vars.experiments_all

def set_log_ax(axis, x, y, style):
    """ Adding diagonal lines in logarithmic axis """
    axis.loglog(x, y,
              color="black",
              linestyle=style,
              alpha=0.5,
              linewidth=0.5)

def scatter_plot(atom, model, ax, ylim, region, title, color_reg, parameters, parameters_reg_loc, exp,right_panel=False):
    """ Used to create a scatter plot per panel """
 #, 'g', 'm', 'k'
    pl = ax.scatter(atom,
                    model,
                    color=color_reg[0],
                    marker=color_reg[1],
                    s = [13 for i in range(len(atom))],
                    label=region)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim((ylim[0], ylim[1]))
    ax.set_xlim((ylim[0], ylim[1]))
    if exp == 'SPMOAon':
        label = 'PNC$_{ATom}$' f'({title[1]} m$^{{-3}}$) \n\n'
    else:
        label = ' '
    ax.set_xlabel(label,
                  fontsize=font)
    ax.set_ylabel('PNC$_{HAM}$' f'({title[1]} m$^{{-3}}$)',
                  fontsize=font)

    k = 10
    set_log_ax(ax,
               ylim,
               ylim,
              "-",)
    set_log_ax(ax,
               ylim,
               [ylim[0]*k, ylim[1] * k],
           ":")
    set_log_ax(ax,
              ylim ,
              [ylim[0]/k, ylim[1]/ k],
                ":")

    ax.set_title(title[0],
                 loc='center',
                 fontsize=font)
    ax.tick_params(axis='both',
                   labelsize=font)

    ax.text(0.1, 0.95,
            parameters,
            verticalalignment='top',
            horizontalalignment='left',
            transform=ax.transAxes,
            color='k',
            fontsize=font-2)

    ax.text(0.1+parameters_reg_loc[1], ylim[0],
            parameters_reg_loc[0],
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            color=color_reg[0],
            fontsize=font-2)

    ax.grid(linestyle='--',
            linewidth=0.4)

    return pl


def create_data_dict(ds_atom_time_mean, sel_dates, region_name, dict_data):
    for exp in experiments:
        dict_data[region_name][exp]['Model'] = {}
        dict_data[region_name][exp]['Observation'] = {}
        for mode in mode_atom:
            c_atom, c_echam_tpxy = read_nc_files_No_conc(ds_atom_time_mean,
                                                         sel_dates,
                                                         region_name,
                                                         exp,
                                                         mode)
            print(c_echam_tpxy.values, c_atom.values)
            dict_data[region_name][exp]['Model'][mode] = c_echam_tpxy
            dict_data[region_name][exp]['Observation'][mode] = c_atom
    return dict_data

def clean_nan(x_val, y_val):
    x_val_notnan, y_val_notnan = [], []
    for i, y in enumerate(y_val):
        if isinstance(y, float) and math.isnan(y):
            pass
        else:
            x_val_notnan.append(x_val[i])
            y_val_notnan.append(y_val[i])
    return np.array(x_val_notnan), np.array(y_val_notnan)

def all_reg_stat(regions, factor, i, exp, mode_atom):
    x_reg = []
    y_reg = []
    for j, reg in enumerate(regions.keys()):
        x = dict_data[reg][exp]['Observation'][mode_atom[i]] * factor
        x_reg.append(x)
        y = dict_data[reg][exp]['Model'][mode_atom[i]] * factor
        y_reg.append(y)
    x_reg_one_list = np.array([float(item) for sublist in x_reg for item in sublist])
    y_reg_one_list = np.array([float(item) for sublist in y_reg for item in sublist])
    x_reg_one_list_clean, y_reg_one_list_clean = clean_nan(x_reg_one_list, y_reg_one_list)
    # print(x_reg_one_list_clean, y_reg_one_list_clean)
    dict_stat = utils.get_statistics_updated(x_reg_one_list_clean, y_reg_one_list_clean)

    print(dict_stat['pval'])
    if dict_stat['pval'] < 0.05:
        sl = dict_stat['slope']
        intr = dict_stat['intercept']
        r = dict_stat['pearsons_coeff']
        eq_linear_reg = f' y = {np.round(sl, 2)}x {np.round(intr, 3)} \n R: {np.round(r, 2)} \n \n'
    else:
        eq_linear_reg = ''
    rmse = dict_stat['RMSE']
    mb = dict_stat['MB']
    nmb = dict_stat['NMB']
    parameters = (f'{eq_linear_reg}'
                  f'RMSE: {np.round(rmse, 2)} \n '
                  f'MB: {np.round(mb, 2)} \n '
                  f'NMB: {np.round(nmb, 2)} \n ')

    return parameters


def plot_fig_thesis(dict_data, mode_atom_double=None):
    fig, axs = plt.subplots(2,2, figsize=(8, 10))
    axs.flatten()
    regions = utils.region_definition()
    color_marker_reg = [['y', 'o'],
                        ['r', 'X'],
                        ['lightgreen', 's']]
    exp_names_plot = ['SPMOAoff', 'SPMOAon']
    ylimits = [[10**6, 10**10],
               [10**5, 10**8]] #[10**6, 10**10],
    factor = [10**-10, 10**-8]#10**-10,
    ax_label = ['10$^{10}$', '10$^{8}$'] #'10$^{10}$',
    locs = [0, 0.3, 0.6]
    mode_atom_names = global_vars.mode_atom_names[1:]
    mode_atom = global_vars.mode_atom_all[1:]

    for e, exp in enumerate(experiments):
        print(exp)
        for i, ax in enumerate(axs[e][:2]):
            if i == 1:
                ax.set_title(rf'{exp_names_plot[0]}'+'\n \n',
                             loc='center',
                             fontsize=font)
            parameters = all_reg_stat(regions, factor[i], i, exp, mode_atom)
            for j, reg in enumerate(regions.keys()):
                x = dict_data[reg][exp]['Observation'][mode_atom[i]]*factor[i]
                y = dict_data[reg][exp]['Model'][mode_atom[i]].values*factor[i]
                title = exp_names_plot[e] + '\n \n' + mode_atom_names[i]
                x_clean, y_clean = clean_nan(x, y)

                dict_stat = utils.get_statistics_updated(x_clean, y_clean)
                rmse = dict_stat['RMSE']
                mb = dict_stat['MB']
                nmb = dict_stat['NMB']
                parameters_reg = (f'RMSE: {np.round(rmse, 2)} \n '
                              f'MB: {np.round(mb, 2)} \n '
                              f'NMB: {np.round(nmb, 2)} \n ')
                param_location = [parameters_reg,
                                  locs[j]]

                pl = scatter_plot(x_clean,
                                  y_clean,
                                 ax,
                                 [v*factor[i] for v in ylimits[i]],
                                 reg,
                                 [title, ax_label[i]],
                                 color_marker_reg[j],
                                 parameters,
                                  param_location,
                                  exp_names_plot[e],
                                 right_panel=False)

    handles, labels = axs[1][-1].get_legend_handles_labels()
    fig.legend(labels=labels, handles=handles,
                      loc='lower center', bbox_to_anchor=(0.5, 0.01),
                      fancybox=True, shadow=True, ncol=3, fontsize=font)
    fig.tight_layout()

    plt.savefig(f'plots/scatter_plot_regions.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    with open(f"Data_dict.pkl", "rb") as myFile:
        dict_data = pickle.load(myFile)

    plot_fig_thesis(dict_data)
