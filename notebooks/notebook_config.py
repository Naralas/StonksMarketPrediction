import seaborn as sns
import matplotlib
import sys


def setup_notebook():
    sys.path.insert(0, '..')
    matplotlib.pyplot.rcParams["figure.figsize"] = (20,8)
    sns.set_theme()
    font = {'family' : 'DejaVu Sans', 'size'   : 25}
    matplotlib.rc('font', **font)
