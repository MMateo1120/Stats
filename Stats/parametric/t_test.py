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
import sys


class _CommonMethods:
    def __init__(self, data1, alpha, type):
        self.data1 = data1
        self.alpha = alpha
        self.type = type        
    
    def residual_plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.tight_layout(pad=3.0)
        fig.suptitle("Residual plots")
        sm.ProbPlot(self.residuals.iloc[:, 0]).qqplot(line="s", ax=ax1)
        sns.histplot(self.residuals, stat="density", legend=False, ax=ax2),
        x = np.linspace(-4 * self.residuals.std(), 4 * self.residuals.std(), 200)
        p = norm.pdf(x, 0, self.residuals.std())
        z = plt.plot(x, p, color="red", linewidth=2)
        plt.show()
    
    def normality(self):
        return {"p-value": shapiro(self.residuals)[1].item()}

    def __init_warnings(self):
        sys.tracebacklimit = 0
        if not isinstance(self.type, str) or self.type not in [
            "one-sided",
            "two-sided",
        ]:
            raise TypeError("The 'type' parameter must be 'one-sided' or 'two-sided'!")

        if not isinstance(self.alpha, float) or self.alpha >= 1 or self.alpha <= 0:
            raise ValueError("The 'alpha' parameter must be a 'float' between 0 and 1!")
        
        if not isinstance(self.data1, pd.DataFrame) or self.data1.shape[1] != 1:
             raise TypeError(f"The 'data1' parameter must be a single column in a Pandas DataFrame!")
         
        if hasattr(self,'data2') and (not isinstance(self.data2, pd.DataFrame) or self.data2.shape[1] != 1):
             raise TypeError(f"The 'data2' parameter must be a single column in a Pandas DataFrame!")
         
        if hasattr(self,'reference') and not isinstance(self.reference, int|float):
             raise TypeError("The 'reference' parameter must be 'float'!")
    
    def __test_warnings(self, direction):
        if self.type in ["one-sided"] and direction is None:
            raise Exception("Argument of .test() must be '>' or '<'!")
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

class OneSample(_CommonMethods):
    def __init__(self, data1: pd.DataFrame, alpha=0.05, type: Literal["one-sided","two-sided"] ="two-sided", reference: float|int =0.0):
        self.reference = reference
        super().__init__(data1, alpha, type)
        
        self._CommonMethods__init_warnings()
 
    def __stats(self):
            self.__mean = self.data1.mean().item()
            self.__ste = self.data1.std().item() / (len(self.data1) ** 0.5)
            self.residuals = self.data1 - self.__mean
            self.__test_statistic = (self.__mean - self.reference) / self.__ste
            self.__df = len(self.data1) - 1
            self.__inverse_t = t.cdf(self.__test_statistic, self.__df)

    def test(self, mean_vs_reference=None):
        self.__stats()
        self._CommonMethods__test_warnings(mean_vs_reference)
        
        results_dict = {
            "mean": [self.__mean],
            "ste": [self.__ste],
            "test_statistic": [self.__test_statistic],
        }

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
                if mean_vs_reference == ">":
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CUL = self.__mean + t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% upper CL", CUL),
                        ("p-value", p_value, 5),
                    ]
                    
                elif mean_vs_reference == "<":
                    p_value = 1 - self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    CLL = self.__mean - t_crit * self.__ste
                    new_results = [
                        (f"{100-100*self.alpha}% lower CL", CLL),
                        ("p-value", p_value, 5),
                    ]
                    
        results_dict.update(new_results)
    
        return pd.DataFrame(results_dict)

    def plot(self):
        sns.boxplot(self.data1)
        plt.axhline(y=self.reference, color="r", linestyle="-")
        plt.show()


class UnpairedSamples(_CommonMethods):
    def __init__(self, data1: pd.DataFrame, data2: pd.DataFrame, alpha=0.05, type: Literal["one-sided","two-sided"] ="two-sided", pooling: bool= True):
        super().__init__(data1,alpha, type)
        self.data2 = data2
        self.pooling = pooling
        

        self._CommonMethods__init_warnings()
    
    def __stats(self):
        self.__mean1 = self.data1.mean()
        self.__mean2 = self.data2.mean()
        self.__std1 = self.data1.std() 
        self.__std2 = self.data2.std() 
        self.__df1 = len(self.data1) - 1 
        self.__df2 = len(self.data2) - 1
        self.residuals = pd.concat([self.data1-self.data1.mean(),self.data2-self.data2.mean()], keys=['res1','res2'])
        
        if self.pooling:
            """Do an F test"""
            self.__df = self.__df1 + self.__df2
            self.__std_pooled = ((self.__df1*self.__std1**2 + self.__df2*self.__std2**2) / self.__df)**0.5
            self.__test_ste = (self.__std_pooled*(1/(len(self.data1))+1/(len(self.data2)))**0.5)
  
        else:
            s1_term = self.__std1**2/len(self.data1)
            s2_term = self.__std2**2/len(self.data2)
            self.__df = (s1_term + s2_term)**2 / ((1/self.__df1)*(s1_term)**2 + (1/self.__df2)*(s2_term)**2)
            self.__test_ste = ((self.__std1**2/(len(self.data1)) + self.__std2**2/(len(self.data2)))**0.5)
            
        self.__test_diff = self.__mean1 - self.__mean2
        self.__test_statistic = self.__test_diff / self.__test_ste
        self.__inverse_t = t.cdf(self.__test_statistic, self.__df)        
            
    def test(self, mean1_vs_mean2=None):
        self.__stats()
        self._CommonMethods__test_warnings(mean1_vs_mean2)
        
        results_dict = {
            "mean1": [self.__mean1.item()],
            "mean2": [self.__mean2.item()],
            "std1": [self.__std1.item()],
            "std2": [self.__std2.item()],
            "diff_means": [self.__test_diff.item()],
            "ste": [self.__test_ste.item()],
            "test_statistic": [self.__test_statistic.item()],
        } 

        match self.type:
            case "two-sided":
                p_value = (1 - self.__inverse_t) * 2
                t_crit = t.ppf(1 - self.alpha / 2, self.__df)
                CUL = self.__test_diff + t_crit * self.__test_ste
                CLL = self.__test_diff - t_crit * self.__test_ste
                new_results = [
                    (f"{100-100*self.alpha}% lower CL", CLL),
                    (f"{100-100*self.alpha}% upper CL", CUL),
                    ("p-value", p_value),
                ]
                
            case "one-sided":
                if mean1_vs_mean2 == ">":
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    #we accept the H0 as long as mean2 < CUL
                    CUL = self.__mean1 + t_crit * self.__test_ste
                    new_results = [
                        (f"{100-100*self.alpha}% upper CL", CUL),
                        ("p-value", p_value),
                    ]
                    
                elif mean1_vs_mean2 == "<":
                    p_value = 1 - self.__inverse_t
                    t_crit = t.ppf(1 - self.alpha, self.__df)
                    #we accept the H0 as long as mean2 > CLL
                    CLL = self.__mean1 - t_crit * self.__test_ste
                    new_results = [
                        (f"{100-100*self.alpha}% lower CL", CLL),
                        ("p-value", p_value),
                    ]
        results_dict.update(new_results)
        results = pd.DataFrame(results_dict)
        
        if self.pooling:
            results.insert(4,'std_pooled',self.__std_pooled)

        return results
    
    def plot(self):
        data_all = pd.concat([pd.concat([self.data1.rename(columns={0: "value"}),pd.DataFrame({'Sample': [1]*len(self.data1)})], axis =1),pd.concat([self.data2.rename(columns={0: "value"}),pd.DataFrame({'Sample': [2]*len(self.data2)})], axis =1)])
        sns.boxplot(data_all,x="Sample",y="value")
        plt.show()
        
        
class PairedSamples(OneSample):
    def __init__(self, data1: pd.DataFrame, data2: pd.DataFrame, alpha=0.05, type: Literal["one-sided","two-sided"] ="two-sided"):
        super().__init__(data1,alpha, type)
        self.data2 = data2
        self.__onesample = OneSample(self.data1-self.data2, self.alpha, self.type)
        
    def test(self, mean1_vs_mean2=None):
        self.__onesample.test()
        self.residuals = self.__onesample.residuals
        result = self.__onesample.test()
        result.rename(columns={'mean': 'means_diff'}, inplace=True)
        return result
    
    def plot(self):
        self.__onesample.plot()
        