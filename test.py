from parametric.t_test import one_sample as os
import importlib
importlib.reload(os)
import random
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as sm 
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats

data = []
for i in range (40):
    data.append(random.gauss(0,1))    
data = pd.DataFrame(data)
asd = os.OneSample(data)
asd.test()
asd.residual_plot()
asd.plot()

import os
os.getcwd()

os.chdir('C:\\Users\\Mihalovits\\Projects\\Stats')




