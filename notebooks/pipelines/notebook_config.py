import seaborn as sns
import matplotlib
import sys
import os

def setup_notebook():
    sys.path.insert(0, '../..')
    matplotlib.pyplot.rcParams["figure.figsize"] = (20,8)
    sns.set_theme()
    os.environ["WANDB_SILENT"]="true"
    font = {'family' : 'DejaVu Sans', 'size'   : 25}
    matplotlib.rc('font', **font)
