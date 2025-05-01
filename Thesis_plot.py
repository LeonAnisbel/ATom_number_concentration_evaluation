import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import math
import global_vars
import utils
from plots import read_nc_files_No_conc

font = 10
def scatter_plot(atom, model, ax, parameter, title, var, right_panel=False):
    """ Used to create a scatter plot per panel """
    color_reg = ['y', 'r', 'lightgreen'] #, 'g', 'm', 'k'
    pl = plt.scatter(atom,
                    model,
                    color=color_reg,
                    ax = ax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    yli_min = 10 ** -5
    ax.set_ylim((yli_min, 10 ** 0))
    ax.set_xlim((yli_min, 10 ** 0))
    ax.set_xlabel('Observation (${\mu}$g m$^{-3}$)',
                  fontsize=font)
    ax.set_ylabel(f'Model' +  '(${\mu}$g m$^{-3}$)',
                  fontsize=font)

    k = 10
    # set_log_ax(ax,
    #            [yli_min, 10 ** 0],
    #            [yli_min, 10 ** 0],
    #           "-",)
    # set_log_ax(ax,
    #            [yli_min , 10 ** 0],
    #            [yli_min*k, 10 ** 0 * k],
    #        ":")
    # set_log_ax(ax,
    #           [yli_min, 10 ** 0] ,
    #           [yli_min/k, 10 ** 0 / k],
    #             ":")

    ax.set_title(f'{title[0]} ({var})',
                 loc='right',
                 fontsize=font)
    ax.set_title(title[1],
                 loc='left',
                 fontsize=font)
    ax.tick_params(axis='both',
                   labelsize=font)

    # ax.text(0.1, 0.95,
    #         parameter,
    #         verticalalignment='top',
    #         horizontalalignment='left',
    #         transform=ax.transAxes,
    #         color='k',
    #         fontsize=font-2)
    ax.legend_.remove()

    ax.grid(linestyle='--',
            linewidth=0.4)

    return pl




def plot_thesis_figure(ds_atom_time_mean, sel_dates, region_name):
    c_echam_base, c_echam_ac3 = [], []
    for exp in ['echam_base', 'ac3_arctic']:
        print(exp)
        c_atom, c_echam_tpxy = read_nc_files_No_conc(ds_atom_time_mean, sel_dates, region_name, exp)
        if exp == 'echam_base':
            c_echam_base.append(c_echam_tpxy)
        else:
            c_echam_ac3.append(c_echam_tpxy)

    fig, axs = plt.subplots(1,3, figsize=(7, 5))
    axs.flatten()
    regions = utils.region_definition()
    for i, ax in enumerate(axs):
        scatter_plot(c_atom,
                     c_echam_base[i],
                     ax,
                     None,
                     regions.keys()[i],
                     'SPMOAoff',
                     right_panel=False)
    plt.savefig(f'plots/scatter_plot_regions.png', dpi=300)
    plt.close()


