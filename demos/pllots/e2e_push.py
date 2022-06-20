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
label = ["$x$", "$y$", "$z$"]
reader = DataReader('./e2epush.mds')

start_time = 5000
end_time = -20000

timesteps = 1e-3*np.arange(0,len(reader.data[names[0]][:,0])+end_time - start_time)


f, ax = plt.subplots(8, 1, sharex=True, figsize=(25,18))

for i in range(3):
    ax[i].plot(timesteps, reader.data[names[3]][:,i][start_time:end_time], label = "cube")
    if i == 0:
        ## 5 cm is added in the x axis to offset the cube size + vicon markers 
        ax[i].plot(timesteps, reader.data[names[4]][:,i][start_time:end_time] + 0.05, label = "end effector")
    else:
        ax[i].plot(timesteps, reader.data[names[4]][:,i][start_time:end_time], label = "end effector")
    ax[i].axvline(x=0, color='k', linestyle='--')
    ax[i].axvline(x=0.2, color='k', linestyle='--')

    ax[i].axvline(x=1.5, color='k', linestyle='--')
    ax[i].axvline(x=1.7, color='k', linestyle='--')
    
    ax[i].axvline(x=3.0, color='k', linestyle='--')
    ax[i].axvline(x=3.2, color='k', linestyle='--')
    
    ax[i].axvline(x=3.9, color='k', linestyle='--')
    ax[i].axvline(x=4.1, color='k', linestyle='--')
    
    ax[i].axvline(x=4.8, color='k', linestyle='--')
    ax[i].axvline(x=5, color='k', linestyle='--')
    
    ax[i].set_ylabel(label[i] + " [m]")
    ax[i].add_patch(Rectangle((0, -.5), 0.2, 2, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((1.5, -.5), 0.2, 2, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((3, -.5), 0.2, 2, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((3.9, -.5), 0.2, 2, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((4.8, -.5), 0.2, 2, color="grey", edgecolor = "black", alpha = 0.4))
    # if i == 0:
    #     ax[i].legend(loc = "lower right")

for i in range(3,8):
    if i < 3+3:
        umax = 2.5
        ax[i].axhline(y=2.5, linestyle='--', color=(1, 0, 0, 0.5), label = "Acceleration limit")
        ax[i].axhline(y=-2.5, color=(1, 0, 0, 0.5), linestyle='--')
    else:
        umax = 1.5
        ax[i].axhline(y=1.5, color=(1, 0, 0, 0.5), linestyle='--', label = "Acceleration limit")
        ax[i].axhline(y=-1.5, color=(1, 0, 0, 0.5), linestyle='--')

    ax[i].add_patch(Rectangle((0, -umax), 0.2, 2*umax, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((1.5, -umax), 0.2, 2*umax, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((3, -umax), 0.2, 2*umax, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((3.9, -umax), 0.2, 2*umax, color="grey", edgecolor = "black", alpha = 0.4))
    ax[i].add_patch(Rectangle((4.8, -umax), 0.2, 2*umax, color="grey", edgecolor = "black", alpha = 0.4))


    ax[i].axvline(x=0, color='k', linestyle='--')
    ax[i].axvline(x=0.2, color='k', linestyle='--')

    ax[i].axvline(x=1.5, color='k', linestyle='--')
    ax[i].axvline(x=1.7, color='k', linestyle='--')
    
    ax[i].axvline(x=3.0, color='k', linestyle='--')
    ax[i].axvline(x=3.2, color='k', linestyle='--')
    
    ax[i].axvline(x=3.9, color='k', linestyle='--')
    ax[i].axvline(x=4.1, color='k', linestyle='--')
    
    ax[i].axvline(x=4.8, color='k', linestyle='--')
    ax[i].axvline(x=5, color='k', linestyle='--')
    

    ax[i].plot(timesteps, reader.data[names[5]][:,i-3][start_time:end_time], label = "Joint Acceleration", color = "green")
    n = i-2
    ax[i].set_ylabel("$a_{}$".format(str(n)) + " [r/$s^{2}$]")
    # if i == 7:
    #     ax[i].legend(loc = "lower right")

handles, labels = ax[0].get_legend_handles_labels()
h2, l2 = ax[5].get_legend_handles_labels()
handles += h2
labels += l2
f.legend(handles, labels, loc='lower right', prop={'size': 20})
f.align_ylabels()

ax[7].set_xlabel("Time [s]")
# ax[2].set_xlabel("Time [s]")
# plt.show()
plt.savefig("e2epush.png", bbox_inches='tight')