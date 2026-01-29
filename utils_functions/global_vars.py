main_dir = '/vols/fs1/work/leon/ATom_vs_ECHAM'
dir_atom = f'{main_dir}/ATom-merge_ncfiles/MER-SAGA'
height_limit = 1000 # height limit in m
experiment = 'echam_base'
experiment = 'ac3_arctic'
dir_echam = f'{main_dir}/ECHAM_HAM_output/{experiment}/NUM_radius'
netcdf_file_dir = '../nc_files/'
plot_dir = '../plots/'
mode_atom = 'coarse' # select the mode
units = 'm$^{{-3}}$'

experiments_all = ['echam_base', 'ac3_arctic']
mode_atom_all = ['fine', 'accumulation', 'coarse']
mode_atom_names = ['fine', 'acc', 'coa']

params_echam = {
    'nucleation': {'sigma_i': 1.59},
    'aitken_i': {'sigma_i': 1.59},
    'aitken_s': {'sigma_i': 1.59},
    'accumulation_i': {'sigma_i': 1.59},
    'accumulation_s': {'sigma_i': 1.59},
    'coarse_i': {'sigma_i': 2},
    'coarse_s': {'sigma_i': 2}
    }
data_echam_number = {
    'nucleation': {'file ending': '*NUM_NUC_plev.nc', 'var. name': 'NUM_NUC'},
    'aitken_i': {'file ending': '*NUM_KI_plev.nc', 'var. name': 'NUM_KI'},
    'aitken_s': {'file ending': '*NUM_KS_plev.nc', 'var. name': 'NUM_KS'},
    'accumulation_i': {'file ending': '*NUM_AI_plev.nc', 'var. name': 'NUM_AI'},
    'accumulation_s': {'file ending': '*NUM_AS_plev.nc', 'var. name': 'NUM_AS'},
    'coarse_i': {'file ending': '*NUM_CI_plev.nc', 'var. name': 'NUM_CI'},
    'coarse_s': {'file ending': '*NUM_CS_plev.nc', 'var. name': 'NUM_CS'}
}
data_echam_rmedian = {
    'nucleation': {'file ending': '*m_radius_NUC_plev.nc', 'var. name': 'rwet_NUC'},
    'aitken_i': {'file ending': '*m_radius_KI_plev.nc', 'var. name': 'rdry_KI'},
    'aitken_s': {'file ending': '*m_radius_KS_plev.nc', 'var. name': 'rwet_KS'},
    'accumulation_i': {'file ending': '*m_radius_AI_plev.nc', 'var. name': 'rdry_AI'},
    'accumulation_s': {'file ending': '*m_radius_AS_plev.nc', 'var. name': 'rwet_AS'},
    'coarse_i': {'file ending': '*m_radius_CI_plev.nc', 'var. name': 'rdry_CI'},
    'coarse_s': {'file ending': '*m_radius_CS_plev.nc', 'var. name': 'rwet_CS'}
}