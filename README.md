# Evaluation of ECHAM6.3-HAM2.3 organic aerosol number concentration against aircraft observations
> The aerosol number concentration from the aerosol-climate model ECHAM6.3-HAM2.3 output are mapped onto predefine aerosol modes from ATom aircraft campaign [here](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1925). \
> It is based on the modelled number concentration and median radius of the simulation organic aerosols.
> 
> Find a more detailed description in Leon-Marcos et al. 2026 (Doctoral Theis at Leipzig University).\
> The code was initially developed by Swetlana Paul (PhD. candidate at TROPOS) and further adapted for regional evaluation by Anisbel Leon Marcos (during the PhD. at TROPOS).
>
> The aerosol-climate model output is archived on Levante HPC system. The netcdf files used in this project are the interpolated fields to vertical pressure levels to facilitate the interpolation to the flight trajectory. 
> See also [these project](https://github.com/LeonAnisbel/transform_vertical_lev_echam_ham_model.git) with bash scripts to create the netcdf files with pressure-level.
> 
> The conda environment for this project is contained in env.yml file. Run [start_env.py](start_env.py) to set up and start the environment for this project.


* Run [main.py](model_atom_comparison/main.py) to read ECHAM-HAM model data and ATom data, perform the integration of model modes over the limits of ATom modes from observations


* Run [Thesis_plot.py](model_atom_comparison/Thesis_plot.py) to create multipanel plot of model vs. Atom number concentration. It also includes subregions in the southern ocean color-coded in the plot with statistics.


