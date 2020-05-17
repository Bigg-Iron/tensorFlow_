# Load necessary libraries
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

# %tensorflow_version 1.x
import tensorflow as tf
from tensorflow.python.data import Dataset

# tf.logging.ERROR and tf.logging.set_verbosity is deprecated
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.ERROR
tf.compat.v1.logging.set_verbosity

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
print (tf.python.data)