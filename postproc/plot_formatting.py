import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import os

tex_width = 483.41216  # pt
tex_font = 10
tex_linewidth = 1
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def plot_formatting(page_ratio, aspect_ratio=4/3):
    """
    Calculate the proper sizes for latex plots
    Args:
    - page_ratio: Desired fraction of the available LaTeX page width to be occupied by the plot
    - aspect_ratio: Desired ratio between plot width and height
    """
    inches_per_pt = 1 / 72.27
    fig_width = page_ratio * tex_width * inches_per_pt
    fig_height = fig_width/aspect_ratio
    fontsize = tex_font / page_ratio
    linewidth = tex_linewidth / page_ratio
    figsize = (fig_width, fig_height)
    return figsize, linewidth

def setup():
    plt.rcParams['axes.prop_cycle'] = cycler(color=custom_colors)
    # Use matplotlib's built-in math rendering with LaTeX style
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('mathtext', fontset='cm')  # Computer Modern fonts
    matplotlib.rc('font', family='serif', size=tex_font)

setup()