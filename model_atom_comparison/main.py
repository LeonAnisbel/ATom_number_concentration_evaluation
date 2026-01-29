import Thesis_plot
from utils_functions import utils, global_vars, read_data, plots
import pickle


regions = utils.region_definition() # get regions definitions in a dictionary
experiments = global_vars.experiments_all # list of experiments ID
dict_data = {}
for reg in regions.keys():
    dict_data[reg] = {}
    dict_data[reg][experiments[0]] = {}
    dict_data[reg][experiments[1]] = {}

    print(reg)
    ds_atom_time_mean = read_data.read_atom(reg) # read ATom data
    data_echam_num, data_echam_rmed, sel_dates = read_data.read_model(ds_atom_time_mean) # read model data
    utils.No_particles_ECHAM_to_ATOM(data_echam_num, data_echam_rmed, reg) # Creates netcdf files with the model modes mapped to ATom modes
    dict_data  = Thesis_plot.create_data_dict(ds_atom_time_mean, sel_dates, reg, dict_data)

    # Create plots (they were not used in any publication just for analysis)
    plots.plot_absolute_diff_map(ds_atom_time_mean, sel_dates, reg)
    plots.scatter_plot(ds_atom_time_mean, sel_dates, reg)
    plots.plot_relat_diff_map(ds_atom_time_mean, sel_dates, reg)
    plots.plot_No_conc_ECHAM(data_echam_rmed, data_echam_num, reg)

with open(f"Data_dict.pkl", "wb") as myFile:
    pickle.dump(dict_data, myFile) # save data for its use later

