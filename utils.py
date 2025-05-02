from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
import math
import global_vars
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import linregress

def No_particles_ECHAM_to_ATOM(data_echam_num, data_echam_rmed, region_name):
    """Calculate number of particles in ECHAM that go into ATom fine, accumulation and coarse, respectively #
"""
    parameters_echam = global_vars.params_echam
    exp = global_vars.experiment

    parts_fine_echam = []
    parts_acc_echam = []
    parts_coa_echam = []

    for mode_echam_k in parameters_echam.keys():
        # r_median_i for every time step
        r_median = data_echam_rmed[mode_echam_k]['reduced dataset'].where(
            data_echam_rmed[mode_echam_k]['reduced dataset'] > 0, other=np.nan)

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
        alpha_atom_fine = (np.log(fine_atom_lower) - np.log(r_median)) / np.log(sigma_mode)
        beta_atom_fine = (np.log(fine_atom_upper) - np.log(r_median)) / np.log(sigma_mode)

        # number of particles in ECHAM inside ATom mode limits (with N = 1)
        N_atom_fine = (norm.cdf(beta_atom_fine) - norm.cdf(alpha_atom_fine))

        # fraction of N_tot that should be counted as ATom fine mode
        fraction_fine = (
                                    r_median / r_median) * N_atom_fine  # = N_atom_fine/1; (r_median/r_median) is just to keep the xarray-dimensions

        # number of particles that should be counted as ATom fine mode
        part_fine_echam_from_mode = fraction_fine * data_echam_num[mode_echam_k]['reduced dataset']
        parts_fine_echam.append(part_fine_echam_from_mode)

        # part of integral that should be in ATom accumulation mode
        # ATom mode limits
        acc_atom_lower = 0.03
        acc_atom_upper = 0.25

        # ATom mode limits taking into account subsitution
        alpha_atom_acc = (np.log(acc_atom_lower) - np.log(r_median)) / np.log(sigma_mode)
        beta_atom_acc = (np.log(acc_atom_upper) - np.log(r_median)) / np.log(sigma_mode)

        # number of particles in ECHAM inside ATom mode limits (with N = 1)
        N_atom_acc = (norm.cdf(beta_atom_acc) - norm.cdf(alpha_atom_acc))

        # fraction of N_tot that should be counted as ATom accumulation mode
        fraction_acc = (
                                   r_median / r_median) * N_atom_acc  # = N_atom_acc/1; (r_median/r_median) is just to keep the xarray-dimensions

        # number of particles that should be counted as ATom accumulation mode
        part_acc_echam_from_mode = fraction_acc * data_echam_num[mode_echam_k]['reduced dataset']
        parts_acc_echam.append(part_acc_echam_from_mode)

        # part of integral that should be in ATom coarse mode
        # ATom mode limits
        coa_atom_lower = 0.25
        coa_atom_upper = 2.4

        # ATom mode limits taking into account subsitution
        alpha_atom_coa = (np.log(coa_atom_lower) - np.log(r_median)) / np.log(sigma_mode)
        beta_atom_coa = (np.log(coa_atom_upper) - np.log(r_median)) / np.log(sigma_mode)

        # number of particles in ECHAM inside ATom mode limits (with N = 1)
        N_atom_coa = (norm.cdf(beta_atom_coa) - norm.cdf(alpha_atom_coa))

        # fraction of N_tot that should be counted as ATom coarse mode
        fraction_coa = (
                                   r_median / r_median) * N_atom_coa  # = N_atom_coa/1; (r_median/r_median) is just to keep the xarray-dimensions

        # number of particles that should be counted as ATom coarse mode
        part_coa_echam_from_mode = fraction_coa * data_echam_num[mode_echam_k]['reduced dataset']
        parts_coa_echam.append(part_coa_echam_from_mode)

        # visualize example n(r) with respective integral ATom mode limits #

        # the integral is calculated a) purely numerically and b) with help of substitution and standard normal cumulative distribution function
        # by plotting, one can check whether they match

        # randomly pick one example distribution
        r_med_example = float(r_median.isel(time=0))
        r_example = np.linspace(0.001,
                                max(r_med_example * 5 + 0.001, coa_atom_upper + 1.01),
                                101)

        # lognormal distribution function
        log_normal_pdf_example = \
            lambda r: ((1 / (np.sqrt(2 * np.pi) * np.log(sigma_mode))) *
                       np.exp(-((np.log(r) - np.log(r_med_example)) ** 2) /
                              (2 * np.log(sigma_mode) ** 2)))
        n_example = log_normal_pdf_example(r_example)

        # a) purely numerical calculation
        N_example_num = []
        for r_i in r_example:  # numerical calculation of function's integral
            N_example_num_i = quad(log_normal_pdf_example, 0, r_i)[0]
            N_example_num.append(N_example_num_i)

        # b) calculation with substitution and standard normal cumulative distribution function
        r_example_subs = ((np.log(r_example) -
                           np.log(r_med_example) - np.log(sigma_mode) ** 2)
                          / np.log(sigma_mode))
        N_example_ncdf = (np.exp(np.log(r_med_example) +
                                 (np.log(sigma_mode) ** 2) / 2) *
                          (norm.cdf(r_example_subs)))

        fig, ax = plt.subplots(1, 1)
        ax.plot(r_example,
                n_example,
                color='cornflowerblue')  # lognormal distribution
        ax.plot(r_example,
                N_example_num,
                color='orange')  # integral of lognormal distribution, calculated purely numerically
        ax.plot(r_example,
                N_example_ncdf,
                color='green')  # integral of lognormal distribution, calculated with standard normal cumulative distribution function

        ax.set_xlim(right=2.7)

        ax.vlines(fine_atom_lower,
                  0,
                  0.9,
                  color='blueviolet')
        ax.text(fine_atom_lower,
                0,
                'fine_atom_lower',
                rotation='vertical')

        ax.vlines(acc_atom_lower,
                  0,
                  0.9,
                  color='mediumvioletred')
        ax.text(acc_atom_lower,
                0.5,
                'acc_atom_lower',
                rotation='vertical')

        ax.vlines(acc_atom_upper,
                  0,
                  0.9,
                  color='firebrick')
        ax.text(acc_atom_upper,
                0,
                'fine/acc to coa',
                rotation='vertical')

        ax.vlines(coa_atom_upper,
                  0,
                  0.9,
                  color='chocolate')
        ax.text(coa_atom_upper,
                0,
                'coa_atom_upper',
                rotation='vertical')

        ax.set_title(
            'ECHAM number distribution for example r_median_i,\nintegral of number distribution\nand ATom mode limits')

        plt.gcf().text(0.11,
                       -(0 + 2) * 0.04,
                       f'part of "fine" in ECHAM from this mode = {float(fraction_fine.isel(time=0)) * 100} %')
        plt.gcf().text(0.11,
                       -(1 + 2) * 0.04,
                       f'part of "accumulation" in ECHAM from this mode = {float(fraction_acc.isel(time=0)) * 100} %')
        plt.gcf().text(0.11,
                       -(2 + 2) * 0.04,
                       f'part of "coarse" in ECHAM from this mode = {float(fraction_coa.isel(time=0)) * 100} %')

        figure_dir = global_vars.plot_dir
        plt.savefig(f'{figure_dir}/number_distribution_for_example_r_median_i_{mode_echam_k}_{region_name}.png')
        plt.close()

    nc_file_dir = global_vars.netcdf_file_dir
    # sum up all contributions to ATom modes and save it in new netcdf-files; this seems to be faster
    c_fine_echam_0 = sum(parts_fine_echam)
    c_fine_echam_ds = c_fine_echam_0.to_dataset(name='c_num')
    c_fine_echam_ds.to_netcdf(f'{nc_file_dir}/{exp}_c_fine_echam_{region_name}.nc')

    c_acc_echam_0 = sum(parts_acc_echam)
    c_acc_echam_ds = c_acc_echam_0.to_dataset(name='c_num')
    c_acc_echam_ds.to_netcdf(f'{nc_file_dir}/{exp}_c_acc_echam_{region_name}.nc')

    c_coa_echam_0 = sum(parts_coa_echam)
    c_coa_echam_ds = c_coa_echam_0.to_dataset(name='c_num')
    c_coa_echam_ds.to_netcdf(f'{nc_file_dir}/{exp}_c_coa_echam_{region_name}.nc')




def statistics(c_atom, c_echam_tpxy, sel_dates):
    # %%
    ################################################
    # Calculating statistics for whole time period #
    ################################################
    print('Start calculating statistics')

    num_timesteps = len(sel_dates)

    # model
    std_model = float(np.nanstd(c_echam_tpxy, ddof=1))  # model data's standard deviation at station location
    mean_model = float(np.nanmean(c_echam_tpxy))  # model data's mean at station location

    # observations
    std_obs = float(np.nanstd(c_atom, ddof=1))  # standard deviation of observed values
    mean_obs = float(np.nanmean(c_atom))  # mean of observed values over time

    # model-observations
    RMSE = float(math.sqrt(np.square(np.subtract(c_atom, c_echam_tpxy)).mean(
        skipna=True)))  # root mean square error btw. model and observations at station location
    mean_bias = np.nanmean(
        np.subtract(c_echam_tpxy, c_atom))  # mean bias btw. model and observation at station location
    normalized_mean_bias = np.nansum(np.subtract(c_echam_tpxy, c_atom)) / np.nansum(
        c_atom)  # normalized mean biases btw. model and observations, at station locations
    pearsons_coeff = np.corrcoef(c_atom, c_echam_tpxy)[0, 1]  # Pearson's correlation coefficient
    ioa = 1 - (np.nansum(np.square(c_atom - c_echam_tpxy))) / (np.nansum(
        np.square(
            np.abs(c_echam_tpxy - np.nanmean(c_atom)) + np.abs(c_atom - np.nanmean(c_atom)))))  # index of agreement
    mean_fractional_bias = np.nanmean(2 * (c_echam_tpxy - c_atom) / (c_echam_tpxy + c_atom))  # mean fractional bias
    mean_fractional_error = np.nanmean(
        2 * np.abs(c_echam_tpxy - c_atom) / (c_echam_tpxy + c_atom))  # mean fractional errors
    m_t_wrt_o_t = c_echam_tpxy.values / c_atom.values  # model values with respect to obs. values
    fac2 = np.count_nonzero(np.logical_and(0.5 <= m_t_wrt_o_t,
                                           m_t_wrt_o_t <= 2)) / num_timesteps  # fraction of complete data pairs where model is within a factor of 2 of observation
    fac10 = np.count_nonzero(np.logical_and(0.1 <= m_t_wrt_o_t,
                                            m_t_wrt_o_t <= 10)) / num_timesteps  # fraction of complete data pairs where model is within a factor of 10 of observation

    # %%
    # units = c_echam_tpxy.attrs['units']

    units = global_vars.units
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
        ['fraction of complete data pairs where model (at station location) is within a factor of 10 of obs.', fac10,
         '']
    ]
    return statistical_quantities


def region_definition():
    reg_data = {'South Atlantic': {'lat': [-90, 0], 'lon': [290, 360]},
                'South Pacific': {'lat': [-90, -23], 'lon': [130, 293]},
                'Central Pacific': {'lat': [-23, 23], 'lon': [130, 293]}, }

    return reg_data


def get_statistics_updated(c_atom, c_echam_txy):
    """Function to compute statistics"""
    # model
    std_model = float(np.nanstd(c_echam_txy, ddof=1))  # model data's standard deviation at station location
    mean_model = float(np.nanmean(c_echam_txy))  # model data's mean at station location

    # observations
    std_obs = float(np.nanstd(c_atom, ddof=1))  # standard deviation of observed values
    mean_obs = float(np.nanmean(c_atom))  # mean of observed values over time

    # model-observations
    # root mean square error btw. model and observations at station location
    MSE = mean_squared_error(c_atom, c_echam_txy)
    RMSE = float(math.sqrt(MSE))

    # mean bias btw. model and observation at station location
    mean_bias = np.nanmean(np.subtract(c_echam_txy, c_atom))
    # print(mean_bias, np.nanmean(c_echam_txy - c_atom))
    # print(c_echam_txy, c_atom)

    # normalized mean biases btw. model and observations, at station locations
    normalized_mean_bias = np.nansum(np.subtract(c_echam_txy, c_atom)) / np.nansum(c_atom)

    # correlation coefficients
    res_lin_reg = linregress(c_atom, c_echam_txy)
    pearsons_coeff = res_lin_reg.rvalue
    pval_corr = res_lin_reg.pvalue

    # index of agreement
    ioa = 1 - (np.nansum(np.square(c_atom - c_echam_txy))) / (
        np.nansum(np.square(np.abs(c_echam_txy - np.nanmean(c_atom)) + np.abs(c_atom - np.nanmean(c_atom)))))

    # Coefficient of Determination-R2 score
    # r2 = r2_score(c_atom, c_echam_txy)

    dict_stat = {'model_std': std_model,
                 'obs_std': std_obs,
                 'RMSE': RMSE,
                 'MB': mean_bias,
                 'NMB': normalized_mean_bias,
                 'pval': pval_corr,
                 'pearsons_coeff': pearsons_coeff,
                 # 'r2':r2,
                 'slope': res_lin_reg.slope,
                 'intercept': res_lin_reg.intercept}
    return dict_stat