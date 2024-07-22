from scipy.stats import ttest_1samp
from Stats.parametric.t_test import one_sample
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import shapiro
from tabulate import tabulate

import numpy as np
import pandas as pd


def test_one_sample():
    data = pd.DataFrame(np.random.normal(5, 1, 10))
    # checking validity using scipy.stats.ttest_1samp module
    assert round(
        one_sample.OneSample(data, reference=4).test()["p-value"].item(), 5
    ) == round(ttest_1samp(data, 4).pvalue.item(), 5)
    assert round(
        one_sample.OneSample(data, reference=4, type="two-sided")
        .test()["p-value"]
        .item(),
        5,
    ) == round(ttest_1samp(data, 4).pvalue.item(), 5)
    assert round(
        one_sample.OneSample(data, reference=0, type="one-sided")
        .test(">")["p-value"]
        .item(),
        5,
    ) == round(ttest_1samp(data, 0, alternative="less").pvalue.item(), 5)
    assert round(
        one_sample.OneSample(data, reference=-10).test("<")["p-value"].item(), 5
    ) == round(ttest_1samp(data, -10, alternative="greater").pvalue.item(), 5)
