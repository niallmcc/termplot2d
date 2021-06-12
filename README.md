# termplot2d

termplot2d.py is a utility for plotting 2d data from netcdf4 file to a 256-colour terminal window. requires python 3.4 or later,xarray+netcdf4.

## installation

Download the termplot2d.py file from this repo.

## running

Specify the path to the netcdf4 input file containing array data, the names of the x and y dimension variabes, and the names of the variable names to plot

Example:

```
python termplot2d.py sst_data.nc lon lat sst
```

The resulting plot is written to the terminal and is sized to fit the terminal by default:

## all options

```
usage: termplot2d.py [-h] [--colour-map COLOUR_MAP] [--nan-colour NAN_COLOUR] [--plot-width PLOT_WIDTH] [--plot-height PLOT_HEIGHT] [--min-value MIN_VALUE] [--max-value MAX_VALUE] [--flip] [--nocheck]
                     input_path x_dimension y_dimension variable_names [variable_names ...]

Utility for plotting 2d data from netcdf4 file to a 256-colour terminal window. Requires xarray+netcdf4.

positional arguments:
  input_path            path to a netcdf4 file
  x_dimension           the dimension to plot on the x-axis
  y_dimension           the dimension to plot on the y-axis
  variable_names        the variable name(s) to plot

optional arguments:
  -h, --help            show this help message and exit
  --colour-map COLOUR_MAP
                        choose the colour map from viridis, magma, plasma, inferno, or rgb (in rgb, specify exactly 3 variable names to provide the r,g and b channel values to create a single plot)
  --missing-colour NAN_COLOUR
                        set the name of a colour to represent missing values
  --plot-width PLOT_WIDTH
                        set width of the plot in characters, by default uses the entire terminal width
  --plot-height PLOT_HEIGHT
                        set height of the plot in characters, by default uses the entire terminal height
  --min-value MIN_VALUE
                        set minimum value on the colour scale
  --max-value MAX_VALUE
                        set the maximum value on the colour scale
  --flip                specify the first rows in the image should appear at the bottom of the plot, not the top
  --nocheck             ignore result of checking if the terminal supports 256 colours
```