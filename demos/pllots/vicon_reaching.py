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
from matplotlib.patches import Rectangle

from matplotlib import image as img

DEFAULT_FONT_SIZE = 25
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook',
                      'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']
DEFAULT_FIGURE_FACE_COLOR = 'white'    # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 25 #DEFAULT_FONT_SIZE
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
reader = DataReader('./vicon_3.mds')

start_time = 2500
end_time = -19000

timesteps = 1e-3*np.arange(start_time,len(reader.data[names[0]][:,0])+end_time)


f, ax = plt.subplots(3, 1, sharex=True, figsize=(25,12))

for i in range(3):
    ax[i].plot(timesteps, reader.data[names[3]][:,i][start_time:end_time], label = "cube")
    
    if i == 0:
        ## 5 cm is added in the x axis to offset the cube size 
        ax[i].plot(timesteps, reader.data[names[4]][:,i][start_time:end_time] + 0.05, label = "end effector")
    else:
        ax[i].plot(timesteps, reader.data[names[4]][:,i][start_time:end_time], label = "end effector")
    
    ax[i].axvline(x=2.5, color='k', linestyle='--')
    ax[i].axvline(x=5, color='k', linestyle='--')
    ax[i].axvline(x=7, color='k', linestyle='--')
    ax[i].axvline(x=7.8, color='k', linestyle='--')
    ax[i].axvline(x=10, color='k', linestyle='--')
    ax[i].axvline(x=12.5, color='k', linestyle='--')
    ax[i].axvline(x=15, color='k', linestyle='--')
    ax[i].axvline(x=16, color='k', linestyle='--')
    ax[i].set_ylabel(label[i] + " [m]")
    ax[i].add_patch(Rectangle((2.5, -.5), 2.5, 2, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((7, -.5), 0.8, 2, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((10, -.5), 2.5, 2, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((15, -.5), 1, 2, color="grey", edgecolor = "black", alpha = 0.4))
    if i == 0:
        ax[i].legend()
ax[2].set_xlabel("Time [s]")
plt.show()
# plt.savefig("vicon.png", bbox_inches='tight')