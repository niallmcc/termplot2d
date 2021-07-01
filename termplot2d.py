# MIT License
#
# Copyright (c) 2021 Niall McCarroll
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import numpy as np
import xarray as xr
import math

import sys

class TermPlotter:

    default_colour_map = ["blue","green","red"]

    default_x = ["lon","longitude","x"]
    default_y = ["lat", "latitude","y"]

    ansi_colours = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                    (192, 192, 192), (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255),
                    (255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0), (0, 0, 95), (0, 0, 135), (0, 0, 175),
                    (0, 0, 215), (0, 0, 255), (0, 95, 0), (0, 95, 95), (0, 95, 135), (0, 95, 175), (0, 95, 215),
                    (0, 95, 255), (0, 135, 0), (0, 135, 95), (0, 135, 135), (0, 135, 175), (0, 135, 215), (0, 135, 255),
                    (0, 175, 0), (0, 175, 95), (0, 175, 135), (0, 175, 175), (0, 175, 215), (0, 175, 255), (0, 215, 0),
                    (0, 215, 95), (0, 215, 135), (0, 215, 175), (0, 215, 215), (0, 215, 255), (0, 255, 0), (0, 255, 95),
                    (0, 255, 135), (0, 255, 175), (0, 255, 215), (0, 255, 255), (95, 0, 0), (95, 0, 95), (95, 0, 135),
                    (95, 0, 175), (95, 0, 215), (95, 0, 255), (95, 95, 0), (95, 95, 95), (95, 95, 135), (95, 95, 175),
                    (95, 95, 215), (95, 95, 255), (95, 135, 0), (95, 135, 95), (95, 135, 135), (95, 135, 175),
                    (95, 135, 215), (95, 135, 255), (95, 175, 0), (95, 175, 95), (95, 175, 135), (95, 175, 175),
                    (95, 175, 215), (95, 175, 255), (95, 215, 0), (95, 215, 95), (95, 215, 135), (95, 215, 175),
                    (95, 215, 215), (95, 215, 255), (95, 255, 0), (95, 255, 95), (95, 255, 135), (95, 255, 175),
                    (95, 255, 215), (95, 255, 255), (135, 0, 0), (135, 0, 95), (135, 0, 135), (135, 0, 175),
                    (135, 0, 215), (135, 0, 255), (135, 95, 0), (135, 95, 95), (135, 95, 135), (135, 95, 175),
                    (135, 95, 215), (135, 95, 255), (135, 135, 0), (135, 135, 95), (135, 135, 135), (135, 135, 175),
                    (135, 135, 215), (135, 135, 255), (135, 175, 0), (135, 175, 95), (135, 175, 135), (135, 175, 175),
                    (135, 175, 215), (135, 175, 255), (135, 215, 0), (135, 215, 95), (135, 215, 135), (135, 215, 175),
                    (135, 215, 215), (135, 215, 255), (135, 255, 0), (135, 255, 95), (135, 255, 135), (135, 255, 175),
                    (135, 255, 215), (135, 255, 255), (175, 0, 0), (175, 0, 95), (175, 0, 135), (175, 0, 175),
                    (175, 0, 215), (175, 0, 255), (175, 95, 0), (175, 95, 95), (175, 95, 135), (175, 95, 175),
                    (175, 95, 215), (175, 95, 255), (175, 135, 0), (175, 135, 95), (175, 135, 135), (175, 135, 175),
                    (175, 135, 215), (175, 135, 255), (175, 175, 0), (175, 175, 95), (175, 175, 135), (175, 175, 175),
                    (175, 175, 215), (175, 175, 255), (175, 215, 0), (175, 215, 95), (175, 215, 135), (175, 215, 175),
                    (175, 215, 215), (175, 215, 255), (175, 255, 0), (175, 255, 95), (175, 255, 135), (175, 255, 175),
                    (175, 255, 215), (175, 255, 255), (215, 0, 0), (215, 0, 95), (215, 0, 135), (215, 0, 175),
                    (215, 0, 215), (215, 0, 255), (215, 95, 0), (215, 95, 95), (215, 95, 135), (215, 95, 175),
                    (215, 95, 215), (215, 95, 255), (215, 135, 0), (215, 135, 95), (215, 135, 135), (215, 135, 175),
                    (215, 135, 215), (215, 135, 255), (215, 175, 0), (215, 175, 95), (215, 175, 135), (215, 175, 175),
                    (215, 175, 215), (215, 175, 255), (215, 215, 0), (215, 215, 95), (215, 215, 135), (215, 215, 175),
                    (215, 215, 215), (215, 215, 255), (215, 255, 0), (215, 255, 95), (215, 255, 135), (215, 255, 175),
                    (215, 255, 215), (215, 255, 255), (255, 0, 0), (255, 0, 95), (255, 0, 135), (255, 0, 175),
                    (255, 0, 215), (255, 0, 255), (255, 95, 0), (255, 95, 95), (255, 95, 135), (255, 95, 175),
                    (255, 95, 215), (255, 95, 255), (255, 135, 0), (255, 135, 95), (255, 135, 135), (255, 135, 175),
                    (255, 135, 215), (255, 135, 255), (255, 175, 0), (255, 175, 95), (255, 175, 135), (255, 175, 175),
                    (255, 175, 215), (255, 175, 255), (255, 215, 0), (255, 215, 95), (255, 215, 135), (255, 215, 175),
                    (255, 215, 215), (255, 215, 255), (255, 255, 0), (255, 255, 95), (255, 255, 135), (255, 255, 175),
                    (255, 255, 215), (255, 255, 255), (8, 8, 8), (18, 18, 18), (28, 28, 28), (38, 38, 38), (48, 48, 48),
                    (58, 58, 58), (68, 68, 68), (78, 78, 78), (88, 88, 88), (98, 98, 98), (108, 108, 108),
                    (118, 118, 118), (128, 128, 128), (138, 138, 138), (148, 148, 148), (158, 158, 158),
                    (168, 168, 168), (178, 178, 178), (188, 188, 188), (198, 198, 198), (208, 208, 208),
                    (218, 218, 218), (228, 228, 228), (238, 238, 238)]

    webcolors = {"lavender": (230, 230, 250), "thistle": (216, 191, 216), "plum": (221, 160, 221), "violet": (
        238, 130, 238), "orchid": (218, 112, 214), "fuchsia": (255, 0, 255), "magenta": (255, 0, 255), "mediumorchid": (
        186, 85, 211), "mediumpurple": (147, 112, 219), "blueviolet": (138, 43, 226), "darkviolet": (
        148, 0, 211), "darkorchid": (153, 50, 204), "darkmagenta": (139, 0, 139), "purple": (128, 0, 128), "indigo": (
        75, 0, 130), "darkslateblue": (72, 61, 139), "slateblue": (106, 90, 205), "mediumslateblue": (
        123, 104, 238), "pink": (255, 192, 203), "lightpink": (255, 182, 193), "hotpink": (255, 105, 180), "deeppink": (
        255, 20, 147), "palevioletred": (219, 112, 147), "mediumvioletred": (199, 21, 133), "lightsalmon": (
        255, 160, 122), "salmon": (250, 128, 114), "darksalmon": (233, 150, 122), "lightcoral": (
        240, 128, 128), "indianred": (205, 92, 92), "crimson": (220, 20, 60), "firebrick": (178, 34, 34), "darkred": (
        139, 0, 0), "red": (255, 0, 0), "orangered": (255, 69, 0), "tomato": (255, 99, 71), "coral": (
        255, 127, 80), "darkorange": (255, 140, 0), "orange": (255, 165, 0), "yellow": (255, 255, 0), "lightyellow": (
        255, 255, 224), "lemonchiffon": (255, 250, 205), "lightgoldenrodyellow": (250, 250, 210), "papayawhip": (
        255, 239, 213), "moccasin": (255, 228, 181), "peachpuff": (255, 218, 185), "palegoldenrod": (
        238, 232, 170), "khaki": (240, 230, 140), "darkkhaki": (189, 183, 107), "gold": (255, 215, 0), "cornsilk": (
        255, 248, 220), "blanchedalmond": (255, 235, 205), "bisque": (255, 228, 196), "navajowhite": (
        255, 222, 173), "wheat": (245, 222, 179), "burlywood": (222, 184, 135), "tan": (210, 180, 140), "rosybrown": (
        188, 143, 143), "sandybrown": (244, 164, 96), "goldenrod": (218, 165, 32), "darkgoldenrod": (
        184, 134, 11), "peru": (205, 133, 63), "chocolate": (210, 105, 30), "saddlebrown": (139, 69, 19), "sienna": (
        160, 82, 45), "brown": (165, 42, 42), "maroon": (128, 0, 0), "darkolivegreen": (85, 107, 47), "olive": (
        128, 128, 0), "olivedrab": (107, 142, 35), "yellowgreen": (154, 205, 50), "limegreen": (50, 205, 50), "lime": (
        0, 255, 0), "lawngreen": (124, 252, 0), "chartreuse": (127, 255, 0), "greenyellow": (173, 255, 47), "springgreen": (
        0, 255, 127), "mediumspringgreen": (0, 250, 154), "lightgreen": (144, 238, 144), "palegreen": (
        152, 251, 152), "darkseagreen": (143, 188, 143), "mediumseagreen": (60, 179, 113), "seagreen": (
        46, 139, 87), "forestgreen": (34, 139, 34), "green": (0, 128, 0), "darkgreen": (0, 100, 0), "mediumaquamarine": (
        102, 205, 170), "aqua": (0, 255, 255), "cyan": (0, 255, 255), "lightcyan": (224, 255, 255), "paleturquoise": (
        175, 238, 238), "aquamarine": (127, 255, 212), "turquoise": (64, 224, 208), "mediumturquoise": (
        72, 209, 204), "darkturquoise": (0, 206, 209), "lightseagreen": (32, 178, 170), "cadetblue": (
        95, 158, 160), "darkcyan": (0, 139, 139), "teal": (0, 128, 128), "lightsteelblue": (176, 196, 222), "powderblue": (
        176, 224, 230), "lightblue": (173, 216, 230), "skyblue": (135, 206, 235), "lightskyblue": (
        135, 206, 250), "deepskyblue": (0, 191, 255), "dodgerblue": (30, 144, 255), "cornflowerblue": (
        100, 149, 237), "steelblue": (70, 130, 180), "royalblue": (65, 105, 225), "blue": (0, 0, 255), "mediumblue": (
        0, 0, 205), "darkblue": (0, 0, 139), "navy": (0, 0, 128), "midnightblue": (25, 25, 112), "white": (
        255, 255, 255), "snow": (255, 250, 250), "honeydew": (240, 255, 240), "mintcream": (245, 255, 250), "azure": (
        240, 255, 255), "aliceblue": (240, 248, 255), "ghostwhite": (248, 248, 255), "whitesmoke": (
        245, 245, 245), "seashell": (255, 245, 238), "beige": (245, 245, 220), "oldlace": (253, 245, 230), "floralwhite": (
        255, 250, 240), "ivory": (255, 255, 240), "antiquewhite": (250, 235, 215), "linen": (
        250, 240, 230), "lavenderblush": (255, 240, 245), "mistyrose": (255, 228, 225), "gainsboro": (
        220, 220, 220), "lightgray": (211, 211, 211), "silver": (192, 192, 192), "darkgray": (169, 169, 169), "gray": (
        128, 128, 128), "dimgray": (105, 105, 105), "lightslategray": (119, 136, 153), "slategray": (
        112, 128, 144), "darkslategray": (47, 79, 79), "black": (0, 0, 0) }

    reset_escape_code = "\u001b[0m"

    def __init__(self,ds,colour_map,missing_colour,x_dimension,y_dimension,plot_width,plot_height,min_value,max_value,flip):
        """
        Create a TerminalPlotter.  Call the plot method of a TerminalPlotter instance to generate plots.

        :param ds: an xarray dataset
        :param colour_map: the name of the colour map to use, either comma separated colour list or "rgb"
        :param missing_colour: name of a colour to represent a missing value
        :param x_dimension: the name of the dimension to plot on the x-axis
        :param y_dimension: the name of the dimension to plot on the y-axis
        :param plot_width: the width of the plot (or None to automatically select)
        :param plot_height: the height of the plot (or None to automatically select)
        :param min_value: the minimum value to plot on the colour scale (or None to automatically select)
        :param plot_height: the maximum value to plot on the colour scale (or None to automatically select)
        :param flip: set to true if the first rows in the image should appear at the bottom of the plot
        """
        self.ds = ds
        self.colour_map = colour_map
        self.cached_colours = {} # mapping from (r,g,b) fractions to the closest ANSI colours
        self.data = None
        self.height = None
        self.width = None
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension

        # if x and y dimensions are not defined, try to guess them
        if self.x_dimension == "":
            for x_dimension in TermPlotter.default_x:
                if x_dimension in self.ds.variables:
                    self.x_dimension = x_dimension
                    break

        if self.y_dimension == "":
            for y_dimension in TermPlotter.default_y:
                if y_dimension in self.ds.variables:
                    self.y_dimension = y_dimension
                    break

        if self.x_dimension == "":
            print("Unable to guess x-dimension, please specify using -x/--x-dimension")
            sys.exit(-1)

        if self.y_dimension == "":
            print("Unable to guess y-dimension, please specify using -y/--y-dimension")
            sys.exit(-1)

        if self.colour_map != "rgb":
            self.compute_colour_scale(32)
        else:
            self.colour_scale = None

        if missing_colour in TermPlotter.webcolors:
            (r,g,b) = TermPlotter.webcolors[missing_colour]
            self.missing_colour_code = self.getClosestColourCode(r,g,b)
        else:
            print("colour to represent missing (%s) not recognized, using black"%(missing_colour))
            self.missing_color_code = 0

        self.nan_fraction = 0.0
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.min_value = min_value
        self.max_value = max_value

        if not self.plot_height or not self.plot_width:
            tsize = os.get_terminal_size()
            if not self.plot_height:
                self.plot_height = tsize.lines - 2
            if not self.plot_width:
                self.plot_width = tsize.columns - 1
        self.flip = flip

    def compute_colour_scale(self,colour_count):
        """
        Compute a graduated colour scale from a list of colours
        :param colour_count: the number of graduations
        """
        colours = self.colour_map.split(",")
        colours_rgb = []
        for colour in colours:
            (r,g,b) = TermPlotter.webcolors.get(colour)
            colours_rgb.append((r/255,g/255,b/255))

        self.colour_scale = []
        for i in range(colour_count):
            frac = i/colour_count
            index = math.floor((len(colours_rgb)-1)*frac)
            (r0,g0,b0) = colours_rgb[index]
            (r1,g1,b1) = colours_rgb[index+1]
            frac = (frac - index/(len(colours_rgb)-1)) * (len(colours_rgb)-1)
            self.colour_scale.append((r0+frac*(r1-r0),g0+frac*(g1-g0),b0+frac*(b1-b0)))

    def plot(self,var_names):
        """
        make a plot of one or more variables
        :param var_names: a list of variable names to plot
        :return: the contents of the plots, concatenated if multiple plots are generated
        """
        if var_names == []:
            # no variables specified, plot all variables with the specified x and y dimensions?
            for var_name in self.ds.variables:
                v = self.ds.variables[var_name]
                if self.x_dimension in v.dims and self.y_dimension in v.dims:
                    var_names.append(var_name)
        plots = []
        for var_name in var_names:
            plots.append(self.plotvar(self.ds,var_name))
        return plots

    def plotvar(self,ds,var_name):
        """
        make a plot of a single variable
        :param ds: an xarray dataset
        :param var_name: the name of a variable within the dataset
        :return: (nan_fraction,minval,maxval,data,original_height,original_width)
        """
        (nan_fraction,minval,maxval,data,original_height,original_width) = self.loadvar(ds,var_name)

        (height, width) = data.shape

        self.cbar = ""
        for index in range(len(self.colour_scale)):
            self.cbar += self.getColourBGString(self.getColourCode(index))
        self.cbar += TermPlotter.reset_escape_code

        s = ""
        for y in range(0, height):
            last_code = None
            for x in range(0, width):
                v = data[y, x]
                if math.isnan(v):
                    code = self.missing_colour_code
                else:
                    index = math.floor(len(self.colour_scale) * v)
                    code = self.getColourCode(index)
                if last_code is not None and code == last_code:
                    s += " "
                else:
                    s += self.getColourBGString(code)
                    last_code = code
            s += TermPlotter.reset_escape_code
            s += "\n"
        s += "%s (w:%d,h:%d) [%f %s %f] [missing: %.3f%% %s]" % (
            var_name, original_width, original_height,
            minval, self.cbar, maxval, 100 * nan_fraction,
            self.getColourBGString(self.missing_colour_code, s=" ", reset=True))
        return s

    def loadvar(self,ds,var_name):
        """
        load and wrangle a variable from the dataset
        :param ds: the xarray dataset
        :param var_name: the name of a variable in the dataset
        :return: (nan_fraction,minval,maxval,data,original_height,original_width)
        """
        variable = ds[var_name]
        dims = variable.dims

        # work out the index of the x and y dimensions in the array
        x_index = dims.index(self.x_dimension)
        y_index = dims.index(self.y_dimension)
        original_height = variable.shape[y_index]
        original_width = variable.shape[x_index]

        # extract a 2D dataset, setting other indices to 0
        lookup = []
        for index in range(len(dims)):
            if index != x_index and index != y_index:
                # for dimensions other than x, and y, use a fixed index
                lookup.append(0)
            elif index == x_index:
                lookup.append(slice(0, original_width))
            elif index == y_index:
                lookup.append(slice(0, original_height))
        arr = variable[tuple(lookup)]

        # get NaN statistics
        nan_fraction = np.count_nonzero(np.isnan(arr.data)) / (original_width * original_height)

        # work out the size of the window to coarsen the array
        window_size_x = math.ceil(ds.sizes[self.x_dimension] / self.plot_width)
        window_size_y = math.ceil(ds.sizes[self.y_dimension] / self.plot_height)

        # if the window size in either dimension is > 1, coarsen the data
        if window_size_x > 1 or window_size_y > 1:
            arr = arr.coarsen({self.y_dimension: window_size_y, self.x_dimension: window_size_x}, boundary="pad")\
                .mean().data

        # work out max and min values if not specified explicitly
        maxval = self.max_value if self.max_value is not None else np.nanmax(arr)
        minval = self.min_value if self.min_value is not None else np.nanmin(arr)

        # normalise the array so that values lie in the range 0.0 to 1.0
        data = (arr - minval) / (maxval - minval)

        # make sure that array is organised by [y,x]
        if y_index > x_index:
            data = np.transpose(data)

        if self.flip:
            data = np.flipud(data)

        return (nan_fraction,minval,maxval,data,original_height,original_width)

    def getColourCode(self,index):
        """
        gets an ansi control code that sets the background colour close to a colour map index
        :param index: the index into the colour scale
        :return: the ansi colour code that most closely matches the index into the colour map
        """
        if index < 0:
            index = 0
        elif index >= len(self.colour_scale):
            index = len(self.colour_scale) - 1
        r, g, b = self.colour_scale[index]
        return self.getClosestColourCode(int(255 * r), int(255 * g), int(255 * b))

    def getColourBGString(self,ansi_colour_code,s=" ",reset=False):
        """
        gets a background coloured string
        :param ansi_colour_code: the ansi colour code, in the range 0 to 255
        :param s: the string to print
        :param reset: whether to reset the colours at the end of the string
        :return: a string which prints the coloured string, using the closest available colour in the ANSI palette
        """
        return "\u001b[48;5;" + str(ansi_colour_code) + "m"+s + (TermPlotter.reset_escape_code if reset else "")

    def getClosestColourCode(self, r, g, b):
        """
        gets an ansi colour code that most closely matches given r,g,b values
        :param r: red value in the range 0 to 255
        :param g: green value in the range 0 to 255
        :param b: blue value in the range 0 to 255
        :return: the ansi colour code that most closely matches the (r,g,b) values
        """
        # to avoid lengthy repeated searches, check if the value has already been computed and cached
        if (r, g, b) in self.cached_colours:
            return self.cached_colours[(r, g, b)]

        # search through the ansi colours to find the most similar one
        closest_sqdistance = None
        closest_index = None
        for col_index in range(len(TermPlotter.ansi_colours)):
            (lr, lg, lb) = TermPlotter.ansi_colours[col_index]
            sqdistance = (lr - r) ** 2 + (lg - g) ** 2 + (lb - b) ** 2
            if closest_sqdistance is None or sqdistance < closest_sqdistance:
                closest_sqdistance = sqdistance
                closest_index = col_index

        # update the cache to avoid recomputation of the same value
        self.cached_colours[(r,g,b)] = closest_index
        return closest_index

    def clearTerminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description="Utility for plotting 2d data from netcdf4 file to a 256-colour terminal window. "+
                    "Requires xarray+netcdf4.")
    parser.add_argument("input_path", help="path to a netcdf4 file")
    parser.add_argument("-x", "--x-dimension", dest="x", metavar="<dimension>",
                help="the dimension to plot on the x-axis",default="")
    parser.add_argument("-y", "--y-dimension", dest="y", metavar="<dimension>",
                help="the dimension to plot on the y-axis",default="")
    parser.add_argument("-v", "--variable", dest="variables",
                help="the variable name(s) to plot", nargs="+", metavar="<variable>", default=[])
    parser.add_argument("--colour-map",
                help="choose the colour map as a comma separated list of colours", default="blue,green,red")
    parser.add_argument("--missing-colour",
                help="set the name of a colour to represent NaN values", default="black")
    parser.add_argument("--plot-width",
                help="set width of the plot in characters, by default uses the entire terminal width",type=int)
    parser.add_argument("--plot-height",
                help="set height of the plot in characters, by default uses the entire terminal height",type=int)
    parser.add_argument("--min-value",
                help="set minimum value on the colour scale",type=float)
    parser.add_argument("--max-value",
                help="set the maximum value on the colour scale",type=float)
    parser.add_argument("--flip",action="store_true",
                help="specify that the first rows in the image should appear at the top of the plot, not the bottom")
    parser.add_argument("--nocheck", action="store_true",
                help="ignore result of checking if the terminal supports 256 colours")


    args = parser.parse_args()

    if os.getenv("TERM") != "xterm-256color":
        print(("WARNING: " if args.nocheck else "ERROR: ") + "terminal does not appear to support 256 colours.")
        if not args.nocheck:
            sys.exit(-1)

    ds = xr.open_dataset(args.input_path)

    tp = TermPlotter(ds,args.colour_map,args.missing_colour,
                     args.x,args.y,
                     args.plot_width,args.plot_height,
                     args.min_value,args.max_value,
                     not args.flip)

    plots = tp.plot(args.variables)

    if len(plots) == 0:
        print("No variables found to plot")

    # print the first plot
    if len(plots) >= 1:
        tp.clearTerminal()
        print(plots[0])

    # print any remaining plots, waiting for key presses
    for x in range(1,len(plots)):
        input("Press Any Key>")
        tp.clearTerminal()
        print(plots[x])



