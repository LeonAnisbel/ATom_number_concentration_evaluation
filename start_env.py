import os
from utils_functions import global_vars

os.system('conda env_evaluation create -f env.yml')
os.system('conda activate env_evaluation')


try:
    os.makedirs(global_vars.plot_dir)
except OSError:
    pass

try:
    os.makedirs(global_vars.netcdf_file_dir)
except OSError:
    pass

