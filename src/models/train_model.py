'''
    4. calculate C-Index from final predictions
    5. set up hyperparam tuning
'''

import numpy as np
import pandas as pd

from rnnsurv.utils import get_data

XT, YT = get_data(nrows=2000)
