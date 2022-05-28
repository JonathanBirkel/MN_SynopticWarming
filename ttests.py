'''
Basic t-test functions for model output stats, callable from wrf_new

All are 20-yr annual series (averages, event counts, etc) 
so may not need to worry about autocorrelation?

'''

import numpy as np
import scipy.stats
import pandas as pd

# New simplified version with parameter flexibility, based on old version below
def ttest_new(sample_1, sample_2, fromStats=True, equalVar=False):
    avg_1 = sample_1.mean()
    avg_2 = sample_2.mean()
    std_1 = sample_1.std(ddof=1)  # For the SciPy t-test we use standard deviations, not variances
    std_2 = sample_2.std(ddof=1)
    # The difference betweeen ttest_ind_from_stats and ttest_ind is removed by setting std(ddof=1)
    # (making it a sample stddev, which is what ttest_ind uses) where the default ddof is 0.
    
    n_raw_1 = len(~np.isnan(sample_1)) # using len instead of count_nonzero as below
    n_raw_2 = len(~np.isnan(sample_2)) # want to keep zero values, esp for air mass freq
    
    if fromStats:
        ttest = scipy.stats.ttest_ind_from_stats(avg_1, std_1, n_raw_1, avg_2, std_2, n_raw_2)
    else:
        ttest = scipy.stats.ttest_ind(sample_1, sample_2, equal_var=equalVar)    
    return ttest



# Older t-test code adapted from Gabriel K-S - not all features may be relevant

def count(series):
    return np.count_nonzero(~np.isnan(series))

def esacr(x, mxlag):
    avg = x.mean()
    var = x.var()
    results = []
    for k in range(mxlag + 1):  # Loop over the lags, from lag-0 to lag-mxlag
        result = 0
        for t in range(len(x) - k):  # Loop over all lag pairs in the input data
            result += (x[t] - avg) * (x[t + k] - avg)
        result /= len(x) - k  # Divide the sum to get the average
        result /= var
        results.append(result)
    return results

def adjusted_dof(series):

    autocorr = esacr(series, 1)[1]

    n = len(~np.isnan(series)) # want to keep zero values, esp for air mass freq
    N = n * (1 - autocorr) / (1 + autocorr)
    if N > n:
        N = n
    return N

def main(sample_1, sample_2):

    avg_1 = sample_1.mean()
    avg_2 = sample_2.mean()
    std_1 = sample_1.std()
    std_2 = sample_2.std()
    n_raw_1 = count(sample_1)
    n_raw_2 = count(sample_2)
    n_adj_1 = adjusted_dof(sample_1)
    n_adj_2 = adjusted_dof(sample_2)
    
    ttest_naive = scipy.stats.ttest_ind_from_stats(avg_1, std_1, n_raw_1, avg_2, std_2, n_raw_2).pvalue
    ttest_adjusted = scipy.stats.ttest_ind_from_stats(avg_1, std_1, n_adj_1, avg_2, std_2, n_adj_2).pvalue

    return ttest_adjusted
    


    
    
    