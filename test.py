import importlib
from parametric.t_test import one_sample as ost
importlib.reload(ost)
import pandas as pd
from scipy.stats import norm
from scipy.stats import stats
from tabulate import tabulate
import random
import matplotlib.pyplot as plt
from scipy.stats import probplot
import numpy as np

data = []
for i in range(100):
    data.append(random.gauss(4,1))

asd = ost.One_sample(pd.DataFrame(data), 4.23, 0.1, 'two-sided')
asd.test()
asd.plot()
asd.residual_plot("hist")



stats.ttest_1samp([3.8, 4, 4.2], 4, alternative = 'greater')

df = pd.DataFrame([1,2,3,4,5])

plt.errorbar(0, 10, 1, fmt='o', linewidth=2, capsize=6)
plt.show()


plt.hist(asd.residuals, density=True, alpha=0.6, color='green')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax)
p = norm.pdf(x, 0, asd.residuals.std())
 
plt.plot(x, p, 'k', linewidth=2, color="red")
 
plt.show()
asd.__df