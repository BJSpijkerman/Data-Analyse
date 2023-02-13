# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:50:50 2023

@author: BJ.Spijkerman

Statistical analysis of magnetic fluxdensity in magnets from Magnit B.V.
and Strongh B.V.

Includes basic statistics:
    
    number of measurements
    smalest measurement
    largests measurement
    range of measurements
    mean
    variance
    skewness
    kurtosis
    median
    mode 
    inter quartlie range
    
Regression between measurements and index

Outliers calculated withing the whiskerplot and the 1.5 IQR criterion

Anderson-Darling test for normality

Coëfficient of variance

confidence bounds/ range for mean and variance calculated by normal dist and chi-squared dist

Bartletts test for equal variance

t-test difference in sample means


"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def belCurve(x):
    mean = np.mean(x)
    std = np.std(x)
    MIN = min(x)
    MAX = max(x)
    x = np.linspace(MIN, MAX, 100)
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    return y_out, x

def calculateOutliers(data, median, IQR, outliers):
    for i in range(len(data)):
        distance_from_median = np.absolute(median - data[i])
        if distance_from_median > 1.5 * IQR:
            outliers.append(data[i])
            
def calculateConfidenceIntervalMean(x, mean, var, q):
    n = len(x)
    stdev = np.sqrt(var)
    T = stats.t.ppf(q = q, df = n-1, loc=mean, scale=stdev)
    upper_bound = mean + T * stdev / np.sqrt(n)
    lower_bound = mean - T * stdev / np.sqrt(n)
    return upper_bound, lower_bound


def calculateCofidenceIntervalVariance(data, var, q):
    k = len(data) - 1
    CHI2_l = stats.chi2.ppf(q = q, df = k)
    CHI2_r = stats.chi2.ppf(q = 1-q, df = k)
    lower_bound = (k * var) / CHI2_r
    upper_bound = (k * var) / CHI2_l
    return upper_bound, lower_bound


# Load data
magnitDat = pd.read_csv('Magnit.B.V.csv', encoding='utf-8')
stronghDat = pd.read_csv('Strongh B.V.csv', encoding='utf-8')

# Seperate data matrices into arrays
magnit = magnitDat['Magnetische fluxdichtheid [T]'].to_numpy(dtype='float')
index_magnit = magnitDat['Meting [#]'].to_numpy(dtype='float')

strongh = stronghDat['Magnetische fluxdichtheid [T]'].to_numpy(dtype='float')
index_strongh = stronghDat['Meting [#]'].to_numpy(dtype='float')

n_magnit = len(magnit)
n_strongh = len(strongh)

# Calculate: min/max, range, mean, variance, skewness, kurtosis, median, mode and inter quartile range
description_magnit = stats.describe(magnit)

number_of_measurements_magnit = description_magnit[0]
min_max_tuple_magnit = description_magnit[1]
min_magnit = min_max_tuple_magnit[0]
max_magnit = min_max_tuple_magnit[1]
range_magnit = max_magnit - min_magnit
mean_magnit = description_magnit[2]
var_magnit = description_magnit[3]
skew_magnit = description_magnit[4]
kurt_magnit = description_magnit[5]
median_magnit = np.median(magnit)
mode_magnit = stats.mode(magnit, keepdims=False)
IQR_magnit = stats.iqr(magnit)


description_strongh = stats.describe(strongh)

number_of_measurements_strongh = description_strongh[0]
min_max_tuple_strongh = description_strongh[1]
min_strongh = min_max_tuple_strongh[0]
max_strongh = min_max_tuple_strongh[1]
range_strongh = max_strongh - min_strongh
mean_strongh = description_strongh[2]
var_strongh = description_strongh[3]
skew_strongh = description_strongh[4]
kurt_strongh = description_strongh[5]
median_strongh = np.median(strongh)
mode_strongh = stats.mode(strongh, keepdims=False)
IQR_strongh = stats.iqr(strongh)


# Calculate regression of data sets vs measurement index
m_magnit, b_magnit, r_magnit, p_magnit, se_magnit = stats.linregress(index_magnit, magnit)
magnit_predict = m_magnit * index_magnit + b_magnit

m_strongh, b_strongh, r_strongh, p_strongh, se_strongh =  stats.linregress(index_strongh, strongh)
strongh_predict = m_strongh * index_strongh + b_strongh


# Calculate outliers from IQR
outliers_magnit = []
calculateOutliers(magnit, median_magnit, IQR_magnit, outliers_magnit)
    
outliers_strongh = []
calculateOutliers(strongh, median_strongh, IQR_strongh, outliers_strongh)


# Test for normal distribution
P_value_magnit = stats.anderson(magnit, dist='norm')
P_value_strongh = stats.anderson(strongh, dist='norm')


# Calculate coëfficient of variance
coefficient_of_variance_magnit = stats.variation(magnit)
coefficient_of_variance_strongh = stats.variation(strongh)


# Calculate confidence interval for mean
upper_bound_mean_magnit, lower_bound_mean_magnit = calculateConfidenceIntervalMean(magnit, mean_magnit, var_magnit, q=0.05)
width_mean_magnit = upper_bound_mean_magnit - lower_bound_mean_magnit

upper_bound_mean_strongh, lower_bound_mean_strongh = calculateConfidenceIntervalMean(strongh, mean_strongh, var_strongh, q=0.05)
width_mean_strongh = upper_bound_mean_strongh - lower_bound_mean_strongh


# Calculate confidence interval for variance
upper_bound_var_magnit, lower_bound_var_magnit = calculateCofidenceIntervalVariance(magnit, var_magnit, q=0.05)
width_var_magnit = upper_bound_var_magnit - lower_bound_var_magnit

upper_bound_var_strongh, lower_bound_var_strongh = calculateCofidenceIntervalVariance(strongh, var_strongh, q=0.05)
width_var_strongh = upper_bound_var_strongh - lower_bound_var_strongh


# Test for equal variance
equal_var = stats.bartlett(magnit, strongh)


# Test if sample means differ significantly
diff_mean = stats.ttest_ind(magnit, strongh, equal_var=False)


# Create plot figure
fig, ([ax, ax1], [ax2, ax3], [ax4, ax5], [ax6, ax7]) = plt.subplots(nrows=4, ncols=2)

# Make boxplots, with 1.5 IQD outliers included
ax.boxplot(magnit)

ax.set_xlabel('Magnit B.V.')
ax.set_ylabel('Magnetisch fluxdichtheid [T]')

ax1.boxplot(strongh)

ax1.set_xlabel('Strongh B.V.')
ax1.set_ylabel('Magnetic flixdensity [T]')


# Create scatter plots for measurement data VS measurementIndex
ax2.plot(index_magnit, magnit_predict, color='#FBB80F')
ax2.scatter(index_magnit, magnit)

ax2.set_xlabel('Meting Magnit B.V.')
ax2.set_ylabel('Magnetisch fluxdichtheid [T]')


ax3.plot(index_strongh, strongh_predict, color='#FBB80F')
ax3.scatter(index_strongh, strongh)

ax3.set_xlabel('Meting Strongh B.V.')
ax3.set_ylabel('Magnetisch fluxdichtheid [T]')


# Calculate Q-Qplot data and plot
QQscatter_magnit, regQQ_magnit = stats.probplot(magnit)
QQpredict_magnit = regQQ_magnit[0] * QQscatter_magnit[0] + regQQ_magnit[1]

ax4.plot(QQscatter_magnit[0], QQpredict_magnit, color='#FBB80F')
ax4.scatter(QQscatter_magnit[0], QQscatter_magnit[1])

ax4.set_xlabel('Theoretische kwartielen Magnit')
ax4.set_ylabel('Steekproef kwartielen Magnit')


QQscatter_strongh, regQQ_strongh = stats.probplot(strongh)
QQpredict_strongh = regQQ_strongh[0] * QQscatter_strongh[0] + regQQ_strongh[1]

ax5.plot(QQscatter_strongh[0], QQpredict_strongh, color='#FBB80F')
ax5.scatter(QQscatter_strongh[0], QQscatter_strongh[1])

ax5.set_xlabel('Theoretische kwartielen Strongh')
ax5.set_ylabel('Steekproef kwartielen Strongh')


# Create histograms
magnitBel, X_magnit = belCurve(magnit)
MAX_magnit = max(magnitBel)
magnitBel = magnitBel * 4 / MAX_magnit

ax6.hist(magnit)
ax6.set_xlabel('Magnit B.V.')
ax6.plot(X_magnit, magnitBel, color='#FBB80F')

stronghBel, X_strongh = belCurve(strongh)
MAX_strongh = max(stronghBel)
stronghBel = stronghBel * 3 / MAX_strongh

ax7.hist(strongh)
ax7.set_xlabel('Strongh B.V.')
ax7.plot(X_strongh, stronghBel, color='#FBB80F')


# Format plot area
plt.suptitle('Statisische plots Magnit B.V. & Strongh B.V. meet data')
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(9, 9, forward=True)