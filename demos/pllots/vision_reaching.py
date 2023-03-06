# plots for vicon reaching with kuka

import time

import struct
import array
import gc

import numpy as np
from mim_data_utils import DataLogger, DataReader

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection

from matplotlib import image as img

DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook',
                      'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']
DEFAULT_FIGURE_FACE_COLOR = 'white'    # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 30 #DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ['pdf', 'png']  # ,'eps']
FIGURES_DPI = 150
SHOW_FIGURES = False
FIGURE_PATH = './'

# axes.hold           : True    # whether to clear the axes by default on
# axes.linewidth      : 1.0     # edge linewidth
# axes.titlesize      : large   # fontsize of the axes title
# axes.color_cycle    : b, g, r, c, m, y, k  # color cycle for plot lines
# xtick.labelsize      : medium # fontsize of the tick labels
# figure.dpi       : 80      # figure dots per inch
# image.cmap   : jet               # gray | jet etc...
# savefig.dpi         : 100      # figure dots per inch
# savefig.facecolor   : white    # figure facecolor when saving
# savefig.edgecolor   : white    # figure edgecolor when saving
# savefig.format      : png      # png, ps, pdf, svg
# savefig.jpeg_quality: 95       # when a jpeg is saved, the default quality parameter.
# savefig.directory   : ~        # default directory in savefig dialog box,
# leave empty to always use current working directory
mpl.rcdefaults()
mpl.rcParams['lines.linewidth'] = DEFAULT_LINE_WIDTH
mpl.rcParams['lines.markersize'] = DEFAULT_MARKER_SIZE
mpl.rcParams['patch.linewidth'] = 1
mpl.rcParams['font.family'] = DEFAULT_FONT_FAMILY
mpl.rcParams['font.size'] = DEFAULT_FONT_SIZE
mpl.rcParams['font.serif'] = DEFAULT_FONT_SERIF
mpl.rcParams['text.usetex'] = DEFAULT_TEXT_USE_TEX
mpl.rcParams['axes.labelsize'] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = DEFAULT_LEGEND_FONT_SIZE
# opacity of of legend frame
mpl.rcParams['legend.framealpha'] = 1.
mpl.rcParams['figure.facecolor'] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
scale = 1.0
mpl.rcParams['figure.figsize'] = 70*scale, 5*scale #23, 18  # 12, 9
# line_styles = 10*['g-', 'r--', 'b-.', 'k:', '^c', 'vm', 'yo']
line_styles = 10*['b',  'c', 'g', 'r', 'y', 'k', 'm']


np.set_printoptions(suppress=True, precision=2)

names = ["joint_positions", "joint_velcoties", "tau_in", "x_des", "ee_pos", "a_des"]
label = ["x", "y", "z"]
reader = DataReader('e2epush.mds')

start_time = 5000
end_time = -20000

timesteps = 1e-3*np.arange(0,len(reader.data[names[0]][:,0])+end_time- start_time)


f, ax = plt.subplots(5, 1, sharex=True, figsize=(25,15))

for i in range(5):
    if i < 3:
        ax[i].axhline(y=2.5, linestyle='--', color=(1, 0, 0, 0.5))
        ax[i].axhline(y=-2.5, color=(1, 0, 0, 0.5), linestyle='--')
    else:
        ax[i].axhline(y=1.5, color=(1, 0, 0, 0.5), linestyle='--')
        ax[i].axhline(y=-1.5, color=(1, 0, 0, 0.5), linestyle='--')
    

    ax[i].plot(timesteps, reader.data[names[5]][:,i][start_time:end_time], label = "Torque")
    ax[i].set_ylabel("$a_{}$".format(str(i)))
    if i == 0:
        ax[i].set_ylabel("$a_{}$".format(str(i)) + " [r/$s^{2}$]")
        ax[i].legend()
ax[4].set_xlabel("Time [s]")


plt.show()

# plt.savefig("vicon.png", bbox_inches='tight')