"""
This module defines a plot style, including font size and colors.
To use it, simply import it:
```
import mrl_analysis.plots.plot_setttings        # Alternative 1
from mrl_analysis.plots.plot_settings import *  # Alternative 2
```

Author(s): 
    Julius Durmann
Contact: 
    julius.durmann@tum.de
Date: 
    2023-03-18

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

plt.style.use('seaborn-v0_8-muted')

font = {
    # 'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 20,
}
mpl.rc('font', **font)
# mpl.rc('axes', prop_cycle=cycler(color=['r', 'g', 'b', 'y']))

# Color keys which can be used for plotting
c_avg = 'steelblue'
c_max = 'chartreuse'
c_min = 'tomato'