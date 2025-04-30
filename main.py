import read_data

ds_atom_time_mean = read_data.read_atom()
data_echam_num, data_echam_rmed, sel_dates = read_data.read_model(ds_atom_time_mean)