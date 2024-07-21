import pandas as pd
from scipy.stats import t
from scipy.stats import shapiro
from scipy.stats import probplot
from scipy.stats import norm
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


class One_sample:
    def __init__(self, data: int, reference=0, alpha=0.05, type="two-sided"):
        self.data = data
        self.reference = reference
        self.alpha = alpha
        self.type = type

        if self.type not in ["one-sided", "two-sided"]:
            raise ValueError("Type must be 'one-sided' or 'two-sided'!")
        
    def __stats(self):
        self.__mean = self.data.mean()[0]
        self.__ste = self.data.std()[0]/(len(self.data)**0.5)
        self.residuals = self.data - self.__mean
        self.__test_statistic = (abs(self.__mean-self.reference)/self.__ste)
        self.__df = len(self.data)-1
        self.__inverse_t = t.cdf(self.__test_statistic, self.__df)
    
    def __warnings(self,mean_vs_reference):
        if self.type in ["one-sided"] and mean_vs_reference is None:
            raise Exception("Mean_vs_reference must be '>' or '<'!")
        if self.normality()["p-value"] < 0.05:
            print(tabulate([[f'The p-value of the Shapiro-Wilk normality test of the residuals is {self.normality()['p-value']}. 
                             Normality of the residuals is not fulfilled, nonparametric test may be preferred.']],
                            ["Warning:"], tablefmt="fancy_grid", stralign = "center") + "\n") 
        

    def test(self, mean_vs_reference=None):
        self.__stats()
        self.__warnings(mean_vs_reference)
        results_dict = {'mean': [self.__mean], 'ste': [self.__ste], 'test_statistic': [self.__test_statistic]}
        
        match self.type:
            case "two-sided":
                p_value = (1-self.__inverse_t)*2
                t_crit = t.ppf(1-self.alpha/2, self.__df)
                CUL = self.__mean + t_crit*self.__ste
                CLL = self.__mean - t_crit*self.__ste                
                new_results = [(f'{100-100*self.alpha}% lower CL', CLL), (f'{100-100*self.alpha}% upper CL', CUL), ('p-value', round(p_value,5))]
                results_dict.update(new_results)
         
            case "one-sided":
                if mean_vs_reference == ">" and self.__mean >= self.reference:
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1-self.alpha, self.__df)
                    CUL = self.__mean + t_crit*self.__ste
                    new_results = [(f'{100-100*self.alpha}% upper CL', CUL), ('p-value', round(p_value,5))]
                    results_dict.update(new_results)
                elif mean_vs_reference == ">" and self.__mean < self.reference:
                    p_value = 1-self.__inverse_t
                    t_crit = t.ppf(1-self.alpha, self.__df)
                    CUL = self.__mean + t_crit*self.__ste 
                    new_results = [(f'{100-100*self.alpha}% upper CL', CUL), ('p-value', round(p_value,5))]
                    results_dict.update(new_results)
                elif mean_vs_reference == "<" and self.__mean >= self.reference:
                    p_value = 1-self.__inverse_t
                    t_crit = t.ppf(1-self.alpha, self.__df)
                    CLL = self.__mean - t_crit*self.__ste 
                    new_results = [(f'{100-100*self.alpha}% lower CL', CLL), ('p-value', round(p_value,5))]
                    results_dict.update(new_results)
                elif mean_vs_reference == "<" and self.__mean < self.reference:
                    p_value = self.__inverse_t
                    t_crit = t.ppf(1-self.alpha, self.__df)
                    CLL = self.__mean - t_crit*self.__ste 
                    new_results = [(f'{100-100*self.alpha}% lower CL', CLL), ('p-value', round(p_value,5))]
                    results_dict.update(new_results)
    
        return pd.DataFrame(results_dict)
        
    def residual_plot(self,type= "hist"):
        match type:
            case "hist": 
                plt.hist(self.residuals, density=True, alpha=0.6, color='green')
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax)
                p = norm.pdf(x, 0, self.residuals.std())
                plt.plot(x, p, 'k', linewidth=2, color="red")
                plt.show()       
            case "npplot":
                fig, ax = plt.subplots()
                probplot(self.residuals.iloc[:,0].tolist(), plot=ax)
                plt.show()
            case _:
                raise Exception("Type of .residual_plot() must be 'hist' or 'npplot'")
         
    def normality(self):
        return {"p-value": shapiro(self.residuals)[1].item()}

    def plot(self):
        plt.boxplot(self.data)
        plt.axhline(y = self.reference, color = 'r', linestyle = '-') 
        plt.show()
