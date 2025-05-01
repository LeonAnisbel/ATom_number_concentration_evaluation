import global_vars
import plots
import read_data
import utils

regions = utils.region_definition()
print(global_vars.experiment)
for reg in regions.keys():
    print(reg)
    ds_atom_time_mean = read_data.read_atom(reg)
    data_echam_num, data_echam_rmed, sel_dates = read_data.read_model(ds_atom_time_mean)
    utils.No_particles_ECHAM_to_ATOM(data_echam_num, data_echam_rmed, reg)

    plots.plot_absolute_diff_map(ds_atom_time_mean, sel_dates, reg)
    plots.scatter_plot(ds_atom_time_mean, sel_dates, reg)
    plots.plot_relat_diff_map(ds_atom_time_mean, sel_dates, reg)
    plots.plot_No_conc_ECHAM(data_echam_rmed, data_echam_num, reg)