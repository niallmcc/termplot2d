import os

import numpy as np
import xarray as xr
import math

import sys

class TermPlotter:

    # Continuous colour maps from matplotlib
    # https://bids.github.io/colormap/
    # https://github.com/BIDS/colormap/blob/master/colormaps.py

    # License regarding the Viridis, Magma, Plasma and Inferno colormaps:
    # New matplotlib colormaps by Nathaniel J. Smith, Stefan van der Walt,
    # and (in the case of viridis) Eric Firing.
    #
    # The Viridis, Magma, Plasma, and Inferno colormaps are released under the
    # CC0 license / public domain dedication. We would appreciate credit if you
    # use or redistribute these colormaps, but do not impose any legal
    # restrictions.
    #
    # To the extent possible under law, the persons who associated CC0 with
    # mpl-colormaps have waived all copyright and related or neighboring rights
    # to mpl-colormaps.
    #
    # You should have received a copy of the CC0 legalcode along with this
    # work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

    simplified_colour_maps = {
        "magma": [(0.001462, 0.000466, 0.013866), (0.013708, 0.011771, 0.068667), (0.039608, 0.031090, 0.133515),
                  (0.074257, 0.052017, 0.202660), (0.113094, 0.065492, 0.276784), (0.159018, 0.068354, 0.352688),
                  (0.211718, 0.061992, 0.418647), (0.265447, 0.060237, 0.461840), (0.316654, 0.071690, 0.485380),
                  (0.366012, 0.090314, 0.497960), (0.414709, 0.110431, 0.504662), (0.463508, 0.129893, 0.507652),
                  (0.512831, 0.148179, 0.507648), (0.562866, 0.165368, 0.504692), (0.613617, 0.181811, 0.498536),
                  (0.664915, 0.198075, 0.488836), (0.716387, 0.214982, 0.475290), (0.767398, 0.233705, 0.457755),
                  (0.816914, 0.255895, 0.436461), (0.863320, 0.283729, 0.412403), (0.904281, 0.319610, 0.388137),
                  (0.937221, 0.364929, 0.368567), (0.960949, 0.418323, 0.359630), (0.976690, 0.476226, 0.364466),
                  (0.986700, 0.535582, 0.382210), (0.992785, 0.594891, 0.410283), (0.996096, 0.653659, 0.446213),
                  (0.997325, 0.711848, 0.488154), (0.996898, 0.769591, 0.534892), (0.995131, 0.827052, 0.585701),
                  (0.992440, 0.884330, 0.640099), (0.989434, 0.941470, 0.697519)],
        "inferno": [(0.001462, 0.000466, 0.013866), (0.013995, 0.011225, 0.071862), (0.042253, 0.028139, 0.141141),
                    (0.081962, 0.043328, 0.215289), (0.129285, 0.047293, 0.290788), (0.183429, 0.040329, 0.354971),
                    (0.238273, 0.036621, 0.396353), (0.290763, 0.045644, 0.418637), (0.341500, 0.062325, 0.429425),
                    (0.391453, 0.080927, 0.433109), (0.441207, 0.099338, 0.431594), (0.491022, 0.117179, 0.425552),
                    (0.540920, 0.134729, 0.415123), (0.590734, 0.152563, 0.400290), (0.640135, 0.171438, 0.381065),
                    (0.688653, 0.192239, 0.357603), (0.735683, 0.215906, 0.330245), (0.780517, 0.243327, 0.299523),
                    (0.822386, 0.275197, 0.266085), (0.860533, 0.311892, 0.230606), (0.894305, 0.353399, 0.193584),
                    (0.923215, 0.399359, 0.155193), (0.946965, 0.449191, 0.115272), (0.965397, 0.502249, 0.073859),
                    (0.978422, 0.557937, 0.034931), (0.985952, 0.615750, 0.025592), (0.987874, 0.675267, 0.065257),
                    (0.984075, 0.736087, 0.129527), (0.974638, 0.797692, 0.206332), (0.960626, 0.859069, 0.298010),
                    (0.947594, 0.917399, 0.410665), (0.954529, 0.965896, 0.540361)],
        "plasma": [(0.050383, 0.029803, 0.527975), (0.132381, 0.022258, 0.563250), (0.193374, 0.018354, 0.590330),
                   (0.248032, 0.014439, 0.612868), (0.299855, 0.009561, 0.631624), (0.350150, 0.004382, 0.646298),
                   (0.399411, 0.000859, 0.656133), (0.447714, 0.002080, 0.660240), (0.494877, 0.011990, 0.657865),
                   (0.540570, 0.034950, 0.648640), (0.584391, 0.068579, 0.632812), (0.625987, 0.103312, 0.611305),
                   (0.665129, 0.138566, 0.585582), (0.701769, 0.174005, 0.557296), (0.736019, 0.209439, 0.527908),
                   (0.768090, 0.244817, 0.498465), (0.798216, 0.280197, 0.469538), (0.826588, 0.315714, 0.441316),
                   (0.853319, 0.351553, 0.413734), (0.878423, 0.387932, 0.386600), (0.901807, 0.425087, 0.359688),
                   (0.923287, 0.463251, 0.332801), (0.942598, 0.502639, 0.305816), (0.959424, 0.543431, 0.278701),
                   (0.973416, 0.585761, 0.251540), (0.984199, 0.629718, 0.224595), (0.991365, 0.675355, 0.198453),
                   (0.994474, 0.722691, 0.174381), (0.993033, 0.771720, 0.154808), (0.986509, 0.822401, 0.143557),
                   (0.974443, 0.874622, 0.144061), (0.956808, 0.928152, 0.152409)],
        "viridis": [(0.267004, 0.004874, 0.329415), (0.277018, 0.050344, 0.375715), (0.282327, 0.094955, 0.417331),
                    (0.282884, 0.135920, 0.453427), (0.278826, 0.175490, 0.483397), (0.270595, 0.214069, 0.507052),
                    (0.258965, 0.251537, 0.524736), (0.244972, 0.287675, 0.537260), (0.229739, 0.322361, 0.545706),
                    (0.214298, 0.355619, 0.551184), (0.199430, 0.387607, 0.554642), (0.185556, 0.418570, 0.556753),
                    (0.172719, 0.448791, 0.557885), (0.160665, 0.478540, 0.558115), (0.149039, 0.508051, 0.557250),
                    (0.137770, 0.537492, 0.554906), (0.127568, 0.566949, 0.550556), (0.120565, 0.596422, 0.543611),
                    (0.120638, 0.625828, 0.533488), (0.132268, 0.655014, 0.519661), (0.157851, 0.683765, 0.501686),
                    (0.196571, 0.711827, 0.479221), (0.246070, 0.738910, 0.452024), (0.304148, 0.764704, 0.419943),
                    (0.369214, 0.788888, 0.382914), (0.440137, 0.811138, 0.340967), (0.515992, 0.831158, 0.294279),
                    (0.595839, 0.848717, 0.243329), (0.678489, 0.863742, 0.189503), (0.762373, 0.876424, 0.137064),
                    (0.845561, 0.887322, 0.099702), (0.926106, 0.897330, 0.104071)]
    }

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

    def __init__(self,colour_map,missing_colour,x_dimension,y_dimension,plot_width,plot_height,min_value,max_value,flip):
        """
        Create a TerminalPlotter.  Call the plot method of a TerminalPlotter instance to generate plots.

        :param colour_map: the name of the colour map to use, one of (viridis,plasma,magma,inferno,rgb)
        :param missing_colour: name of a colour to represent a missing value
        :param x_dimension: the name of the dimension to plot on the x-axis
        :param y_dimension: the name of the dimension to plot on the y-axis
        :param plot_width: the width of the plot (or None to automatically select)
        :param plot_height: the height of the plot (or None to automatically select)
        :param min_value: the minimum value to plot on the colour scale (or None to automatically select)
        :param plot_height: the maximum value to plot on the colour scale (or None to automatically select)
        :param flip: set to true if the first rows in the image should appear at the bottom of the plot
        """
        self.colour_map = colour_map
        self.cached_colours = {} # mapping from (r,g,b) fractions to the closest ANSI colours
        self.data = None
        self.height = None
        self.width = None
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.map_colour_count = len(TermPlotter.simplified_colour_maps[self.colour_map]) if self.colour_map != "rgb" else 0

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
                self.plot_height = tsize.lines - 1
            if not self.plot_width:
                self.plot_width = tsize.columns - 1
        self.flip = flip

    def plot(self,ds,var_names):
        """
        make a plot of one or more variables
        :param ds: an xarray dataset
        :param var_names: a list of variable names to plot
        :return: the contents of the plots, concatenated if multiple plots are generated
        """
        if self.colour_map == "rgb":
            return self.plotrgb(ds,var_names)

        plots = []
        for var_name in var_names:
            plots.append(self.plotvar(ds,var_name))
        return "\n\n".join(plots)

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
        for index in range(self.map_colour_count):
            self.cbar += self.getColouredString(self.getColourCode(index))
        self.cbar += TermPlotter.reset_escape_code

        s = ""
        for y in range(0, height):
            last_code = None
            for x in range(0, width):
                v = data[y, x]
                if math.isnan(v):
                    code = self.missing_colour_code
                else:
                    index = math.floor(self.map_colour_count * v)
                    if index < 0:
                        index = 0
                    elif index >= self.map_colour_count:
                        index = self.map_colour_count - 1
                    code = self.getColourCode(index)
                if last_code is not None and code == last_code:
                    s += " "
                else:
                    s += self.getColouredString(code)
                    last_code = code
            s += TermPlotter.reset_escape_code
            s += "\n"
        s += "\n"
        s += "%s (w:%d,h:%d) [%f %s %f] [missing: %.3f%% %s]" % (
            var_name, original_width, original_height,
            minval, self.cbar, maxval, 100 * nan_fraction,
            self.getColouredString(self.missing_colour_code, s=" ", reset=True))
        return s

    def plotrgb(self,ds,var_names):
        """
        make an RGB plot using three variables to control the red,green and blue intensities separately

        :param ds: the xarray dataset
        :param var_names: a list of three variable names [red-variable,green-variable,blue-variable]
        :return: string containing the plotted data
        """
        channels = []
        for var_name in var_names:
            (nan_fraction, minval, maxval, data, original_height, original_width) = self.loadvar(ds,var_name)
            channels.append((nan_fraction, minval, maxval, data, original_height, original_width))

        (height, width) = data.shape

        # quantize the colour levels in each channel to make it faster to find equivalent ansi colours
        quantize = lambda x: int(x*20)/20

        s = ""
        for y in range(0, height):
            last_code = None
            for x in range(0, width):
                rv = channels[0][3][y, x]
                gv = channels[1][3][y, x]
                bv = channels[2][3][y, x]

                if math.isnan(rv) or math.isnan(gv) or math.isnan(bv):
                    code = self.missing_colour_code
                else:
                    code = self.getClosestColourCode(255*quantize(rv),255*quantize(gv),255*quantize(bv))
                if last_code is not None and code == last_code:
                    s += " " # continue using the current colour
                else:
                    s += self.getColouredString(code)
                    last_code = code
            s += TermPlotter.reset_escape_code
            s += "\n"
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
            arr = arr.coarsen({self.y_dimension: window_size_y, self.x_dimension: window_size_x}, boundary="pad").mean().data

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
        :param index: the index into the selected colour map
        :return: the ansi colour code that most closely matches the index into the colour map
        """
        r, g, b = TermPlotter.simplified_colour_maps[self.colour_map][index]
        return self.getClosestColourCode(int(255 * r), int(255 * g), int(255 * b))

    def getColouredString(self,ansi_colour_code,s=" ",reset=False):
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


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Utility for plotting 2d data from netcdf4 file to a 256-colour terminal window.  Requires xarray+netcdf4.")
    parser.add_argument("input_path", help="path to a netcdf4 file")
    parser.add_argument("x_dimension", help="the dimension to plot on the x-axis")
    parser.add_argument("y_dimension", help="the dimension to plot on the y-axis")
    parser.add_argument("variable_names", help="the variable name(s) to plot", nargs="+")
    parser.add_argument("--colour-map",
                        help="choose the colour map from viridis, magma, plasma, inferno, or rgb (in rgb, specify exactly 3 variable names to provide the r,g and b channel values to create a single plot)", default="viridis")
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
    parser.add_argument("--flip",action="store_true",help="specify the first rows in the image should appear at the bottom of the plot, not the top")
    parser.add_argument("--nocheck", action="store_true",
                        help="ignore result of checking if the terminal supports 256 colours")


    args = parser.parse_args()

    if os.getenv("TERM") != "xterm-256color":
        print(("WARNING: " if args.nocheck else "ERROR: ") + "terminal does not appear to support 256 colours.")
        if not args.nocheck:
            sys.exit(-1)


    ds = xr.open_dataset(args.input_path)

    tp = TermPlotter(args.colour_map,args.missing_colour,args.x_dimension,args.y_dimension,args.plot_width,args.plot_height,args.min_value,args.max_value,args.flip)
    print(tp.plot(ds,args.variable_names))


