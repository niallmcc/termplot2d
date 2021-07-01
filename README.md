# termplot2d

termplot2d.py is a utility for plotting 2d data from netcdf4 file to a 256-colour terminal window. requires python 3.4 or later,xarray+netcdf4.

termplot2d.py is intended to be used to get a quick idea of a 2d dataset's distribution, in remote environments where conventional plots are difficult to generate or view.

## Installation

Download the termplot2d.py file from this repo.  If xarray and netcdf4 libraries are not installed, install them using:

```
pip install xarray
pip install netcdf4
```

## Running

* Specify the path to the netcdf4 input file containing array data.
* [Optional] Use -x <dimension> and -y <dimension> to specify the names of the x and y dimension variabes if these are not obviously named (eg "lat", "latitude", "y")
* [Optional] Specify the names of one or more variables to plot using -v <name>.  If not specified, will plot suitable variables on successive pages.

### Examples 

```
python termplot2d.py 20160101120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc
```

![termplot2d](https://user-images.githubusercontent.com/58978249/121785364-53c4fa00-cbb1-11eb-9a00-7c5241011e42.png)


The resulting plots are written to the terminal and is sized to fit the terminal by default.  Press any key to move to the next plot.

To plot a single variable, use `-v` or `--variable`

```
python termplot2d.py 20160101120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc -v analysed_sst
```

To control the dimension names, use `-x` and `-y`

```
python termplot2d.py 20160101120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc -v analysed_sst -x lon -y lat
```

To specify a custom colour map, provide a list of colours

```
python termplot2d.py 20160101120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc -v analysed_sst -x lon -y lat --colour-map "purple,orange,green"
```

Also:

* If the plot appears upside down, add the `--flip` option
* By default, missing (NaN) values will be plotted in black.  Use `--missing-colour` to use a different colour.

## All options

```
usage: termplot2d.py [-h] [-x <dimension>] [-y <dimension>]
                     [-v <variable> [<variable> ...]]
                     [--colour-map COLOUR_MAP]
                     [--missing-colour MISSING_COLOUR]
                     [--plot-width PLOT_WIDTH] [--plot-height PLOT_HEIGHT]
                     [--min-value MIN_VALUE] [--max-value MAX_VALUE] [--flip]
                     [--nocheck]
                     input_path

Utility for plotting 2d data from netcdf4 file to a 256-colour terminal
window. Requires xarray+netcdf4.

positional arguments:
  input_path            path to a netcdf4 file

optional arguments:
  -h, --help            show this help message and exit
  -x <dimension>, --x-dimension <dimension>
                        the dimension to plot on the x-axis
  -y <dimension>, --y-dimension <dimension>
                        the dimension to plot on the y-axis
  -v <variable> [<variable> ...], --variable <variable> [<variable> ...]
                        the variable name(s) to plot
  --colour-map COLOUR_MAP
                        choose the colour map as a comma separated list of
                        colours
  --missing-colour MISSING_COLOUR
                        set the name of a colour to represent NaN values
  --plot-width PLOT_WIDTH
                        set width of the plot in characters, by default uses
                        the entire terminal width
  --plot-height PLOT_HEIGHT
                        set height of the plot in characters, by default uses
                        the entire terminal height
  --min-value MIN_VALUE
                        set minimum value on the colour scale
  --max-value MAX_VALUE
                        set the maximum value on the colour scale
  --flip                specify that the first rows in the image should appear
                        at the top of the plot, not the bottom
  --nocheck             ignore result of checking if the terminal supports 256
                        colours
```
