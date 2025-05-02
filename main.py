import Thesis_plot
import global_vars
import plots
import read_data
import utils
import pickle

from global_vars import experiment

regions = utils.region_definition()
experiments = global_vars.experiments_all
dict_data = {}
for reg in regions.keys():
    dict_data[reg] = {}
    dict_data[reg][experiments[0]] = {}
    dict_data[reg][experiments[1]] = {}

    print(reg)
    ds_atom_time_mean = read_data.read_atom(reg)
    data_echam_num, data_echam_rmed, sel_dates = read_data.read_model(ds_atom_time_mean)
    #utils.No_particles_ECHAM_to_ATOM(data_echam_num, data_echam_rmed, reg)
    dict_data  = Thesis_plot.create_data_dict(ds_atom_time_mean, sel_dates, reg, dict_data)

    # plots.plot_absolute_diff_map(ds_atom_time_mean, sel_dates, reg)
    # plots.scatter_plot(ds_atom_time_mean, sel_dates, reg)
    # plots.plot_relat_diff_map(ds_atom_time_mean, sel_dates, reg)
    # plots.plot_No_conc_ECHAM(data_echam_rmed, data_echam_num, reg)

with open(f"Data_dict.pkl", "wb") as myFile:
    pickle.dump(dict_data, myFile)

