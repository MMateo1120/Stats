from Stats.parametric import t_test as tt
import importlib

importlib.reload(tt)
from scipy.stats import norm
from scipy import stats
from scipy.stats import shapiro
from tabulate import tabulate

import numpy as np
import pandas as pd

# pytest Stats\parametric\tests\test_t_test.py


def test_OneSample():
    # checking validity using scipy.stats.ttest_1samp module

    data = pd.DataFrame(np.random.normal(np.random.choice([-10, 0, 10]), 1, 10))

    assert round(
        tt.OneSample(data, reference=4).test()["p-value"].item(), 5
    ) == round(stats.ttest_1samp(data, 4).pvalue.item(), 5)

    assert round(
        tt.OneSample(data, reference=4, type="two-sided")
        .test()["p-value"]
        .item(),
        5,
    ) == round(stats.ttest_1samp(data, 4).pvalue.item(), 5)

    assert round(
        tt.OneSample(data, reference=-10, type="one-sided")
        .test(">")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_1samp(data, -10, alternative="less").pvalue.item(), 5
    )

    assert round(
        tt.OneSample(data, reference=-10, type="one-sided")
        .test("<")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_1samp(
            data, popmean=-10, alternative="greater"
        ).pvalue.item(),
        5,
    )


def test_UnpairedSamples():
    # checking validity using scipy.stats.ttest_ind module

    data1 = pd.DataFrame(np.random.normal(np.random.choice([-5, 0, 5]), 2, 10))
    data2 = pd.DataFrame(np.random.normal(np.random.choice([-5, 0, 5]), 2, 15))

    # pooled std
    assert round(
        tt.UnpairedSamples(data1, data2).test()["p-value"].item(), 5
    ) == round(stats.ttest_ind(data1, data2).pvalue.item(), 5)

    assert round(
        tt.UnpairedSamples(data1, data2).test()["p-value"].item(),
        5,
    ) == round(stats.ttest_ind(data1, data2).pvalue.item(), 5)

    assert round(
        tt.UnpairedSamples(data1, data2, type="one-sided")
        .test(">")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_ind(data1, data2, alternative="less").pvalue.item(), 5
    )

    assert round(
        tt.UnpairedSamples(data1, data2, type="one-sided")
        .test("<")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_ind(data1, data2, alternative="greater").pvalue.item(), 5
    )

    # not pooled std
    assert round(
        tt.UnpairedSamples(data1, data2, pooling=False)
        .test()["p-value"]
        .item(),
        5,
    ) == round(stats.ttest_ind(data1, data2, equal_var=False).pvalue.item(), 5)

    assert round(
        tt.UnpairedSamples(data1, data2, pooling=False)
        .test()["p-value"]
        .item(),
        5,
    ) == round(stats.ttest_ind(data1, data2, equal_var=False).pvalue.item(), 5)

    assert round(
        tt.UnpairedSamples(data1, data2, type="one-sided", pooling=False)
        .test(">")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_ind(
            data1, data2, alternative="less", equal_var=False
        ).pvalue.item(),
        5,
    )

    assert round(
        tt.UnpairedSamples(data1, data2, type="one-sided", pooling=False)
        .test("<")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_ind(
            data1, data2, alternative="greater", equal_var=False
        ).pvalue.item(),
        5,
    )


def test_PairedSamples():
    # checking validity using scipy.stats.ttest_ind module

    data1 = pd.DataFrame(np.random.normal(np.random.choice([-5, 0, 5]), 2, 10))
    data2 = pd.DataFrame(np.random.normal(np.random.choice([-5, 0, 5]), 2, 10))

    assert round(
        tt.PairedSamples(data1, data2).test()["p-value"].item(), 5
    ) == round(stats.ttest_rel(data1, data2).pvalue.item(), 5)

    assert round(
        tt.PairedSamples(data1, data2).test()["p-value"].item(),
        5,
    ) == round(stats.ttest_rel(data1, data2).pvalue.item(), 5)

    assert round(
        tt.PairedSamples(data1, data2, type="one-sided")
        .test(">")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_rel(data1, data2, alternative="less").pvalue.item(), 5
    )

    assert round(
        tt.PairedSamples(data1, data2, type="one-sided")
        .test("<")["p-value"]
        .item(),
        5,
    ) == round(
        stats.ttest_rel(data1, data2, alternative="greater").pvalue.item(), 5
    )
