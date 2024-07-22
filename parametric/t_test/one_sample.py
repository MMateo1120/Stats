from scipy.stats import norm
from scipy.stats import t
from scipy.stats import shapiro
from tabulate import tabulate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.graphics.gofplots as sm


class OneSample:
    def __init__(self, data: int, reference=0, alpha=0.05, type="two-sided"):
        self.data = data
        self.reference = reference
        self.alpha = alpha
        self.type = type
        self.__init_warnings()

    def __init_warnings(self):
        if not isinstance(self.type, str) or self.type not in [
            "one-sided",
            "two-sided",
        ]:
            raise TypeError("Type must be 'one-sided' or 'two-sided'!")

        if not isinstance(self.alpha, float) or self.alpha >= 1 or self.alpha <= 0:
            raise ValueError("Alpha must be a 'float' between 0 and 1!")

    def __test_warnings(self, mean_vs_reference=None):
        if self.type in ["one-sided"] and mean_vs_reference is None:
            raise Exception("Mean_vs_reference must be '>' or '<'!")
        if self.normality()["p-value"] < 0.05:
            print(
                tabulate(
                    [
                        [
                            f"The p-value of the Shapiro-Wilk normality test of the residuals is {self.normality()['p-value']}. Normality of the residuals is not fulfilled, nonparametric test may be preferred."
                        ]
                    ],
                    ["Warning:"],
                    tablefmt="fancy_grid",
                    stralign="center",
                )
                + "\n"
            )

    def test(self, mean_vs_reference=None):
        self.__stats()
        self.__test_warnings(mean_vs_reference)
        results_dict = {
            "mean": [self.__mean],
            "ste": [self.__ste],
            "test_statistic": [self.__test_statistic],
        }

        match self.type:
            case "two-sided":
                p_value = (1 - self.__inverse_t) * 2
                t_crit = t.ppf(1 - self.alpha / 2, self.__df)
                CUL = self.__mean + t_crit * self.__ste
                CLL = self.__mean - t_crit * self.__ste
                new_results = [
                    (f"{100-100*self.alpha}% lower CL", CLL),
                    (f"{100-100*self.alpha}% upper CL", CUL),
                    ("p-value", p_value),
                ]
                results_dict.update(new_results)

            case "one-sided":
                if mean_vs_reference == ">" and self.__mean >= self.reference:
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CUL = self.__mean + t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% upper CL", CUL),
                        ("p-value", round(p_value, 5)),
                    ]
                    results_dict.update(new_results)
                elif mean_vs_reference == ">" and self.__mean < self.reference:
                    p_value = 1 - self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CUL = self.__mean + t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% upper CL", CUL),
                        ("p-value", round(p_value, 5)),
                    ]
                    results_dict.update(new_results)
                elif mean_vs_reference == "<" and self.__mean >= self.reference:
                    p_value = 1 - self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CLL = self.__mean - t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% lower CL", CLL),
                        ("p-value", round(p_value, 5)),
                    ]
                    results_dict.update(new_results)
                elif mean_vs_reference == "<" and self.__mean < self.reference:
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CLL = self.__mean - t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% lower CL", CLL),
                        ("p-value", round(p_value, 5)),
                    ]
                    results_dict.update(new_results)

        return pd.DataFrame(results_dict)

    def __stats(self):
        self.__mean = self.data.mean().item()
        self.__ste = self.data.std().item() / (len(self.data) ** 0.5)
        self.residuals = self.data - self.__mean
        self.__test_statistic = abs(self.__mean - self.reference) / self.__ste
        self.__df = len(self.data) - 1
        self.__inverse_t = t.cdf(self.__test_statistic, self.__df)

    def residual_plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout(pad=2.0)
        fig.suptitle("Residual plots")
        sm.ProbPlot(self.residuals.iloc[:, 0]).qqplot(line="s", ax=ax1)
        sns.histplot(self.residuals, stat="density", legend=False, ax=ax2),
        x = np.linspace(-4 * self.residuals.std(), 4 * self.residuals.std(), 200)
        p = norm.pdf(x, 0, self.residuals.std())
        z = plt.plot(x, p, color="red", linewidth=2)
        plt.show()

    def normality(self):
        return {"p-value": shapiro(self.residuals)[1].item()}

    def plot(self):
        sns.boxplot(self.data)
        plt.axhline(y=self.reference, color="r", linestyle="-")
        plt.show()
