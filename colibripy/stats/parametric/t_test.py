"""
t-tests including:
one-sample t-test:  OneSample()
two-sample t-test:  UnpairedSamples()
paired t-test:      PairedSamples()
"""

from scipy.stats import norm
from scipy.stats import t
from scipy.stats import shapiro
from tabulate import tabulate
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.graphics.gofplots as sm

__version__ = "0.1.1"

class _CommonMethods:
    """Base (non-callable) class consisting of the shared methods and parameters of the t-tests' classes."""

    def __init__(
        self,
        sample_1: pd.DataFrame,
        alpha: float = 0.05,
        type: str = "two-sided",
    ):
        """Constructs the object used during the evaluation.

        Parameters
        ----------
        sample_1 : pd.DataFrame
            Contains the data of a sample in a column matrix
        alpha : float, optional
            The significance level used to create confidence limits, by default 0.05
        type : str, optional
            The type of the nullhypothesis, by default "two-sided"
        """

        self.sample_1 = sample_1
        self.alpha = alpha
        self.type = type

    def residual_plot(self):
        """Constructs two residual plots:
        normal probability plot and histogram of the `residuals`
        with the fitted normal density function.

        Only available after `.test()` is called.
        """

        if not isinstance(self.residuals, pd.DataFrame):
            return "Call .test() first to create residuals."
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.tight_layout(pad=3.0)
            fig.suptitle("Residual plots")
            sm.ProbPlot(self.residuals.iloc[:, 0]).qqplot(line="s", ax=ax1)
            sns.histplot(self.residuals, stat="density", legend=False, ax=ax2),
            x = np.linspace(
                -4 * self.residuals.std(), 4 * self.residuals.std(), 200
            )
            p = norm.pdf(x, 0, self.residuals.std())
            z = plt.plot(x, p, color="red", linewidth=2)
            plt.show()

    def normality(self) -> dict:
        """Perform Shapiro Wilk test to check normality of the `residuals`.
        Automatic message generated in `.test()` if p-value < 0.05.

        Only available after `.test()` is called.

        Returns
        -------
        dict
            The p-value of the Shapiro Wilk normality test.
            Callable using `.normality()['p-value']`
        """
        if not isinstance(self.residuals, pd.DataFrame):
            return "Call .test() first to create residuals."
        else:
            return {"p-value": shapiro(self.residuals)[1]}

    def __init_warnings(self):
        """Raises warnings during initialization of the object, if arguments are not valid.

        Raises
        ------
        TypeError
            TypeError if 'type' argument is not 'two-sided' or 'one-sided'.
        ValueError
            ValueError if 'alpha' argument is not a 'float' or not between 0 and 1.
        TypeError
            TypeError if 'sample_1' argument is not a single column in a Pandas DataFrame.
        TypeError
            TypeError if 'sample_2' argument is not a single column in a Pandas DataFrame.
        TypeError
            TypeError if 'reference' argument is not a 'float'.
        """

        if not isinstance(self.type, str) or self.type not in [
            "one-sided",
            "two-sided",
        ]:
            raise TypeError(
                "The 'type' parameter must be 'one-sided' or 'two-sided'!"
            )

        if (
            not isinstance(self.alpha, float)
            or self.alpha >= 1
            or self.alpha <= 0
        ):
            raise ValueError(
                "The 'alpha' parameter must be a 'float' between 0 and 1!"
            )

        if (
            not isinstance(self.sample_1, pd.DataFrame)
            or self.sample_1.shape[1] != 1
        ):
            raise TypeError(
                "The 'sample_1' parameter must be a single column in a Pandas DataFrame!"
            )

        if hasattr(self, "sample_2") and (
            not isinstance(self.sample_2, pd.DataFrame)
            or self.sample_2.shape[1] != 1
        ):
            raise TypeError(
                "The 'sample_2' parameter must be a single column in a Pandas DataFrame!"
            )

        if hasattr(self, "reference") and not isinstance(
            self.reference, int | float
        ):
            raise TypeError("The 'reference' parameter must be 'float'!")

    def __test_warnings(self, direction: str):
        """Raises warnings when test() of the object is called and the arguments are not valid,
        and raises automatic message if normality of the residuals are not fulfilled.

        Parameters
        ----------
        direction : str
            Defined in the method where this function is called.

        Raises
        ------
        ValueError
            ValueError if `type` is `'one-sided'` and no direction is defined.
        """

        if self.type in ["one-sided"] and direction is None:
            raise ValueError("Argument of .test() must be '>' or '<'!")

        if self.normality()["p-value"] < 0.05:
            message = (
                f"The p-value of the Shapiro-Wilk normality test of the residuals is {self.normality()['p-value']}."
                + f" Normality of the residuals is not fulfilled, nonparametric test may be preferred."
            )
            print(
                tabulate(
                    [[message]],
                    ["Warning:"],
                    tablefmt="fancy_grid",
                    stralign="center",
                )
                + "\n"
            )


class OneSample(_CommonMethods):
    """Initializes the evaluation of the one-sample t-test."""

    def __init__(
        self,
        sample_1: pd.DataFrame,
        alpha=0.05,
        type: Literal["one-sided", "two-sided"] = "two-sided",
        reference: float | int = 0.0,
    ):
        """One-sample t-test.

        OneSample compares the population mean of `sample_1` to a `reference` value.

        Parameters
        ----------
        sample_1 : pd.DataFrame
            The data of the sample in a column matrix.
        alpha : float, optional
            The significance level used to create confidence limits,
            by default 0.05.
        type : "one-sided" or "two-sided", optional
            The type of the nullhypothesis, by default "two-sided".
        reference : float | int, optional
            The expected value to which the sample mean is compared,
            by default 0.0.

        Attributes
        ----------
        residuals : pd.DataFrame
            Values of the residuals. Only available after `.test()` is called.
        """

        super().__init__(sample_1, alpha, type)
        self.reference = reference
        self.residuals = "Only available after .test() is called"

        # check for input errors in the parameters:
        self._CommonMethods__init_warnings()

    def __stats(self):
        """Calculates the `residuals` and the statistics needed to perform the test.
        Private class, cannot be called by user. Called automatically in `.test()`.
        """
        self.__mean = self.sample_1.mean().item()
        self.__ste = self.sample_1.std().item() / (len(self.sample_1) ** 0.5)
        self.residuals = self.sample_1 - self.__mean
        self.__test_statistic = (self.__mean - self.reference) / self.__ste
        self.__df = len(self.sample_1) - 1
        self.__inverse_t = t.cdf(self.__test_statistic, self.__df)

    def test(
        self, sample_vs_reference: Literal["<", ">"] = None
    ) -> pd.DataFrame:
        """Performs the one-sample t-test on `sample_1`.

        Parameters
        ----------
        sample_vs_reference : '<' or '>', optional
            Relation of the expected value of `sample_1` and the
            `reference` value in the nullhypothesis, by default None.

            Required only if `type` is `'one-sided'`.

            - `'<'` means that in the nullhypothesis,
            the expected value of `sample_1` is <= than the `reference`.
            - `'>'` means that in the nullhypothesis,
            the expected value of `sample_1` is >= than the `reference`.


        Returns
        -------
        pd.DataFrame
            Contains the results of the test and some calculated statistics. Values callable as `.test()['...']`:
            - `['mean']`: the mean of the sample
            - `['ste']`: the standard error of the sample mean
            - `['test_statistic']`: the calculated test statistic
            - `['(1-alpha)% lower CL']`: (1-`alpha`) percent
              lower confidence limit of `sample_1`'s expected value
            - `['(1-alpha)% upper CL']`: (1-`alpha`) percent
              upper confidence limit of `sample_1`'s expected value
            - `['p-value']`: the p-value of the test

            If the test is `'one-sided'` only one of the confidence limits is available.

            If `type` is `'one-sided'` and `.test('<')` is called, the lower confidence limit is
            returned. It gives the smallest value that if assigned to the `reference`,
            the nullhypothesis is on the verge of acceptance (p-value = `alpha`) when
            `sample_1` is unchanged.

            If `type` is `'one-sided'` and `.test('>')` is called, the upper confidence limit is
            returned. It gives the biggest value that if assigned to the `reference`,
            the nullhypothesis is on the verge of acceptance (p-value = `alpha`) when
            `sample_1` is unchanged.
        """

        # calculate statistics like mean, std, etc. needed for the test:
        self.__stats()

        # check for input errors, like missing '<' or '>' in the case of 'one-sided' test:
        self._CommonMethods__test_warnings(sample_vs_reference)

        # initiate the dictionary containing some statistics that will be returned:
        results_dict = {
            "mean": [self.__mean],
            "ste": [self.__ste],
            "test_statistic": [self.__test_statistic],
        }

        # confidence limits and p-value calculation based on the type ('one-sided' vs 'two-sided')
        match self.type:
            case "two-sided":
                p_value = min((1 - self.__inverse_t) * 2, self.__inverse_t * 2)
                t_crit = t.ppf(1 - self.alpha / 2, self.__df)
                CUL = self.__mean + t_crit * self.__ste
                CLL = self.__mean - t_crit * self.__ste
                new_results = [
                    (f"{100-100*self.alpha}% lower CL", CLL),
                    (f"{100-100*self.alpha}% upper CL", CUL),
                    ("p-value", p_value),
                ]

            case "one-sided":
                if sample_vs_reference == ">":
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CUL = self.__mean + t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% upper CL", CUL),
                        ("p-value", p_value),
                    ]

                elif sample_vs_reference == "<":
                    p_value = 1 - self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CLL = self.__mean - t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% lower CL", CLL),
                        ("p-value", p_value),
                    ]

        results_dict.update(new_results)

        return pd.DataFrame(results_dict)

    def plot(self) -> None:
        """Displays the boxplot of `sample_1`'s data. The red line represents the `reference`"""
        sns.boxplot(self.sample_1)
        plt.axhline(y=self.reference, color="r", linestyle="-")
        plt.show()


class UnpairedSamples(_CommonMethods):
    """Initializes the evaluation of two-sample t-test with **independent** samples."""

    def __init__(
        self,
        sample_1: pd.DataFrame,
        sample_2: pd.DataFrame,
        alpha: float | int = 0.05,
        type: Literal["one-sided", "two-sided"] = "two-sided",
        pooling: bool = True,
    ):
        """Two-sample t-test with **independent** samples.

        UnpairedSamples tests whether two independent samples (`sample_1` and `sample_2`)
        have the same expected value.

        Parameters
        ----------
        sample_1 : pd.DataFrame
            Contains the data of sample_1 in a column matrix.
        sample_2 : pd.DataFrame
            Contains the data of sample_2  in a column matrix.
        alpha : float, optional
            The significance level used to create confidence limits, by default 0.05.
        type : "one-sided" or "two-sided", optional
            The type of the nullhypothesis, by default "two-sided".
        pooling : bool, optional
            If true, uses the pooled standard deviation of the samples, by default True.

        Attributes
        ----------
        residuals : pd.DataFrame
            Values of the residuals from both `sample_1` and `sample_2`. Only available after `.test()` is called.
        """

        super().__init__(sample_1, alpha, type)
        self.sample_2 = sample_2
        self.pooling = pooling
        self.residuals = "Only available after `.test()` is called"

        # check for input errors in the parameters:
        self._CommonMethods__init_warnings()

    def __stats(self):
        """Calculates the residuals and the statistics needed to perform the test.
        Private class, cannot be called by user. Called automatically in test()
        """
        self.__mean1 = self.sample_1.mean()
        self.__mean2 = self.sample_2.mean()
        self.__std1 = self.sample_1.std()
        self.__std2 = self.sample_2.std()
        self.__df1 = len(self.sample_1) - 1
        self.__df2 = len(self.sample_2) - 1
        self.residuals = pd.concat(
            [
                self.sample_1 - self.sample_1.mean(),
                self.sample_2 - self.sample_2.mean(),
            ],
            keys=["res1", "res2"],
        )

        # calculate the test statistics with pooled standard deviation:
        if self.pooling:
            # TODO: Add F-test to check equality of variances
            self.__df = self.__df1 + self.__df2
            self.__std_pooled = (
                (self.__df1 * self.__std1**2 + self.__df2 * self.__std2**2)
                / self.__df
            ) ** 0.5
            self.__test_ste = (
                self.__std_pooled
                * (1 / (len(self.sample_1)) + 1 / (len(self.sample_2))) ** 0.5
            )

        # calculate the test statistics with non-pooled standard deviations,
        # using Welch's method:
        else:
            s1_term = self.__std1**2 / len(self.sample_1)
            s2_term = self.__std2**2 / len(self.sample_2)
            self.__df = (s1_term + s2_term) ** 2 / (
                (1 / self.__df1) * (s1_term) ** 2
                + (1 / self.__df2) * (s2_term) ** 2
            )
            self.__test_ste = (
                self.__std1**2 / (len(self.sample_1))
                + self.__std2**2 / (len(self.sample_2))
            ) ** 0.5

        self.__test_diff = self.__mean1 - self.__mean2
        self.__test_statistic = self.__test_diff / self.__test_ste
        self.__inverse_t = t.cdf(self.__test_statistic, self.__df)

    def test(
        self, sample1_vs_sample2: Literal["<", ">"] = None
    ) -> pd.DataFrame:
        """Performs the two-sample t-test on `sample_1` and `sample_2`.

        Parameters
        ----------
        sample1_vs_sample2 : '<' or '>', optional
            Relation of `sample_1` and `sample_2` in the nullhypothesis, by default None.
            When the evaluation is initiated as `UnpairedSamples(x,y,...)`, the earlier argument `x`
            is `sample_1` and the latter `y` is `sample_2`.

            Required only if `type` is `'one-sided'`.

                - `'<'` means that in the nullhypothesis,
                the expected value of `sample_1` is <= than that of `sample_2`.
                - `'>'` means that in the nullhypothesis,
                the expected value of `sample_1` is >= than that of `sample_2`.

        Returns
        -------
        pd.DataFrame
            Contains the results of the test and some calculated statistics.
            Values callable as `.test()['...']`:
            - `['mean1']`: the mean of `sample_1`
            - `['mean2']`: the mean of `sample_2`
            - `['std1']`: the standard deviation in `sample_1`
            - `['std2']`: the standard deviation in `sample_2`
            - `['diff_means']`: the difference between the sample means
            - `['ste']`: the standard error of the difference between the sample means
            - `['test_statistic']`: the calculated test statistic
            - `['(1-alpha)% lower CL']`: (1-`alpha`) % lower confidence limit for the difference
              between the sample means
            - `['(1-alpha)% upper CL']`: (1-`alpha`) % upper confidence limit for the difference
              between the sample means
            - `['p-value']`: the p-value of the test
            - `['std_pooled']`: pooled standard deviation, only available if `pooling` = `True`

            If the test is `'one-sided'` only one of the confidence limits is available.

            If `type` is `'one-sided'` and `.test('<')` is called, the lower confidence limit is
            returned. It gives the smallest value that if assigned as the mean of `sample_2`,
            the nullhypothesis is on the verge of acceptance (`'p-value'` = `alpha`) when
            `sample_1` is unchanged.

            If `type` is `'one-sided'` and `.test('>')` is called, the upper confidence limit is
            returned. It gives the biggest value that if assigned as the mean of `sample_2`,
            the nullhypothesis is on the verge of acceptance (p-value = `alpha`) when
            `sample_1` is unchanged.
        """

        # calculate statistics like mean, std, etc. needed for the test:
        self.__stats()

        # check for input errors, like missing '<' or '>' in the case of 'one-sided' test:
        self._CommonMethods__test_warnings(sample1_vs_sample2)

        # initiate the dictionary containing some statistics that will be returned:
        self.results_dict = {
            "mean1": [self.__mean1.item()],
            "mean2": [self.__mean2.item()],
            "std1": [self.__std1.item()],
            "std2": [self.__std2.item()],
            "diff_means": [self.__test_diff.item()],
            "ste": [self.__test_ste.item()],
            "test_statistic": [self.__test_statistic.item()],
        }

        # confidence limits and p-value calculation based on the type ('one-sided' vs 'two-sided')
        match self.type:
            case "two-sided":
                p_value = min((1 - self.__inverse_t) * 2, self.__inverse_t * 2)
                t_crit = t.ppf(1 - self.alpha / 2, self.__df)
                CUL = self.__test_diff + t_crit * self.__test_ste
                CLL = self.__test_diff - t_crit * self.__test_ste
                new_results = [
                    (f"{100-100*self.alpha}% lower CL", CLL),
                    (f"{100-100*self.alpha}% upper CL", CUL),
                    ("p-value", p_value),
                ]

            case "one-sided":
                if sample1_vs_sample2 == ">":
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CUL = self.__mean1 + t_crit * self.__test_ste
                    new_results = [
                        (f"{100-100*self.alpha}% upper CL", CUL),
                        ("p-value", p_value),
                    ]

                elif sample1_vs_sample2 == "<":
                    p_value = 1 - self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CLL = self.__mean1 - t_crit * self.__test_ste
                    new_results = [
                        (f"{100-100*self.alpha}% lower CL", CLL),
                        ("p-value", p_value),
                    ]

        self.results_dict.update(new_results)
        results = pd.DataFrame(self.results_dict)

        # add the pooled standard deviation if pooling = True is used
        if self.pooling:
            results.insert(4, "std_pooled", self.__std_pooled)

        return results

    def plot(self) -> None:
        """Displays the boxplots of `sample_1`'s and `sample_2`'s data."""
        self.sample_1 = self.sample_1.rename(columns={0: "value"})
        self.sample_2 = self.sample_2.rename(columns={0: "value"})
        self.sample_1 = pd.concat(
            [
                self.sample_1,
                pd.DataFrame({"Sample": [1] * len(self.sample_1)}),
            ],
            axis=1,
        )
        self.sample_2 = pd.concat(
            [
                self.sample_2,
                pd.DataFrame({"Sample": [2] * len(self.sample_2)}),
            ],
            axis=1,
        )
        data_all = pd.concat(
            [
                self.sample_1,
                self.sample_2,
            ]
        )
        sns.boxplot(data_all, x="Sample", y="value")
        plt.show()


class PairedSamples(OneSample):
    """Initializes the evaluation of two-sample t-test with **dependent** samples,
    that is the paired t-test."""

    def __init__(
        self,
        sample_1: pd.DataFrame,
        sample_2: pd.DataFrame,
        alpha=0.05,
        type: Literal["one-sided", "two-sided"] = "two-sided",
    ):
        """Two-sample t-test with **dependent** samples, that is paired t-test.

        PairedSamples tests whether two dependent samples (`sample_1` and `sample_2`)
        have the same expected value.

        Parameters
        ----------
        sample_1 : pd.DataFrame
            Contains the data of sample_1 in a column matrix.
        sample_2 : pd.DataFrame
            Contains the data of sample_2  in a column matrix.
        alpha : float, optional
            The significance level used to create confidence limits, by default 0.05.
        type : "one-sided" or "two-sided", optional
            The type of the nullhypothesis, by default "two-sided".

        Attributes
        ----------
        residuals: pd.DataFrame
            Values of the residuals. Calculated using the pairwise differences and their mean.
            Only available after `.test()` is called.
        """

        super().__init__(sample_1, alpha, type)
        self.sample_2 = sample_2
        self.residuals = "Only available after `.test()` is called"

        if len(sample_1) != len(sample_2):
            raise Exception("Samples must have the same number of data")

        # create an attribute, which is a class for a one-sample t-test,
        # using the paired differences:
        self.__onesample = OneSample(
            self.sample_1 - self.sample_2, self.alpha, self.type
        )

    def test(
        self, sample1_vs_sample2: Literal["<", ">"] = None
    ) -> pd.DataFrame:
        """Performs the paired t-test on the `sample_1` and `sample_2`.

        Parameters
        ----------
        sample1_vs_sample2 : '<' or '>', optional
            Relation of sample_1 and sample_2 in the nullhypothesis, by default None.
            When the evaluation is initiated as `PairedSamples(x,y,...)`, the earlier argument `x`
            is `sample_1` and the latter `y` is `sample_2`.

            Required only if `type` is `'one-sided'`.
            - `'<'` means that in the nullhypothesis,
            the expected value of `sample_1` is <= than that of `sample_2`.
            - `'>'` means that in the nullhypothesis,
            the expected value of `sample_1` is >= than that of `sample_2`.

        Returns
        -------
        pd.DataFrame
            Contains the results of the test and some calculated statistics. Values callable as `.test()['...']`:
            - `['mean_diffs']`: the mean of the differences between the corresponding paired values
            - `['ste']`: the standard error of mean differences
            - `['test_statistic']`: the calculated test statistic
            - `['(1-alpha)% lower CL']`: (1-alpha) percent lower confidence limit of the mean differences
            - `['(1-alpha)% upper CL']`: (1-alpha) percent upper confidence limit of the mean differences
            - `['p-value']`: the p-value of the test

            If the test is 'one-sided' only one of the confidence limits is available.

            If `type` is `'one-sided'` and `.test('<')` is called, the lower confidence limit is
            returned. It gives the value that if added to the mean of `sample_2`,
            the nullhypothesis is on the verge of acceptance (p-value = `alpha`).

            If `type` is `'one-sided'` and `.test('>')` is called, the upper confidence limit is
            returned. It gives the value that if added to the mean of `sample_2`,
            the nullhypothesis is on the verge of acceptance (p-value = `alpha`).
        """

        self.__onesample.test(sample1_vs_sample2)
        self.residuals = self.__onesample.residuals

        result = self.__onesample.test(sample1_vs_sample2)
        result.rename(columns={"mean": "means_diff"}, inplace=True)
        return result

    def plot(self):
        """Displays the boxplot of the differences between the corresponding values
        from `sample_1` and `sample_2`."""
        self.__onesample.plot()
