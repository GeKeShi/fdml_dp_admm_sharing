# -*- coding:utf-8 -*-
# Plot template for high quality figure ploting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as P
from matplotlib.ticker import FormatStrFormatter
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np 
import os

## Plot font settings
# mtplotlib parameters
fontsize = 30
# P.rcParams['font.family'] = 'serif'
P.rcParams['text.usetex'] = True
# P.rcParams['font.monospace'] = 'Ubuntu Mono'
P.rcParams['font.size'] = fontsize
P.rcParams['axes.labelsize'] = fontsize
P.rcParams['axes.labelweight'] = 'bold'
P.rcParams['axes.titlesize'] = fontsize
P.rcParams['xtick.labelsize'] = fontsize
P.rcParams['ytick.labelsize'] = fontsize
P.rcParams['legend.fontsize'] = fontsize
# P.rcParams['figure.titlesize'] = 28

def _get_cdf_samples(val, mre_range):
    """
    Get cdf sample values of val on min:gap:max positions.
    """
    val = ECDF(val)
    val = val(mre_range)
    return val

def plot_cdf_curves(vals, path_figure, line_labels, line_colors, line_styles, line_widths, line_markers, x_label, xlim):
    GAP = 0.001 # precisioin on x-axis
    # preprocess the data to draw
    vals_x = []
    vals_y = []
    for each_val in vals:
        val_x = np.arange(0, np.max(each_val), GAP)
        val_y = _get_cdf_samples(each_val, val_x)
        vals_x.append(val_x)
        vals_y.append(val_y)

    # draw normal curves for the cdf data
    y_label = "CDF"
    plot_plain_curves(vals_x, vals_y, path_figure, line_labels, line_colors, line_styles, line_widths,line_markers, x_label, y_label, xlim)

def plot_box_plot(vals, path_figure, line_labels, line_colors, line_styles, line_widths, x_label, y_label, figsize=(12,6)):
    fig = P.figure(figsize=figsize)
    ax = P.subplot(1,1,1)
    box = P.boxplot(vals, notch=True, patch_artist=True, showfliers=False, labels=line_labels) # Legend isn't working with patch objects.. 
    for patch, color in zip(box['boxes'], line_colors):
        patch.set_facecolor(color)
    # P.xlabel(x_label)
    P.ylabel(y_label)
    P.grid()
    P.savefig(path_figure, bbox_inches='tight')

def plot_plain_curves(vals_x, vals_y, path_figure, line_labels, line_colors, line_styles, line_widths, line_markers, x_label, y_label, xlim, ylim, figsize=(6,6), xticks=None, yticks=None, is_y_log=False, is_x_log=False):
    # fig = P.figure(figsize=(12,6))
    fig = P.figure(figsize=figsize)
    ax = P.subplot(1,1,1)
    i = 0
    while i < len(vals_x):
        ax.plot(vals_x[i], vals_y[i], linewidth=line_widths[i], linestyle=line_styles[i],
                color=line_colors[i], label=line_labels[i], marker=line_markers[i])
        i = i + 1
    P.xlabel(x_label)
    P.ylabel(y_label)
    P.grid()
    P.legend(loc='best', numpoints=1, frameon=False)
    ax.set_xlim(xlim)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    if not ylim==None:
    	ax.set_ylim(ylim)
    if not xticks==None:
        P.xticks(xticks)
    if not yticks==None:
        P.yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if True==is_y_log:
        ax.set_yscale('log')
    if True==is_x_log:
        ax.set_xscale('log')    
    P.savefig(path_figure, bbox_inches='tight')
    # P.show()

## Possible color
#  Alias	Color
# 'b'	blue
# 'g'	green
# 'r'	red
# 'c'	cyan
# 'm'	magenta
# 'y'	yellow
# 'k'	black
# 'w'	white

## Makers 
# https://matplotlib.org/api/markers_api.html

