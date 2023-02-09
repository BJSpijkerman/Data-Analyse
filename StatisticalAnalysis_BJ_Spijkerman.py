# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:50:50 2023

@author: BJ.Spijkerman

Statistical analysis of magnetic fluxdensity in magnets from Magnit B.V.
and Strongh B.V.
"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def calculateOutliers(data, median, IQR, outliers):
    for i in range(len(data)):
        distance_from_median = np.absolute(median - data[i])
        if distance_from_median > 1.5 * IQR:
            outliers.append(data[i])


# Load data
magnitDat = pd.read_csv('Magnit.B.V.csv', encoding='utf-8')
stronghDat = pd.read_csv('Strongh B.V.csv', encoding='utf-8')

# Seperate data matrices into arrays
magnit = magnitDat['Magnetische fluxdichtheid [T]'].to_numpy(dtype='float')
index_magnit = magnitDat['Meting [#]'].to_numpy(dtype='float')

strongh = stronghDat['Magnetische fluxdichtheid [T]'].to_numpy(dtype='float')
index_strongh = stronghDat['Meting [#]'].to_numpy(dtype='float')


# Calculate: min/max, mean, variance, skewness, kurtosis, median, mode and inter quartile range
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
median_magnit = np.median(magnit, keepdims=False)
mode_magnit = stats.mode(magnit)
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
median_strongh = np.median(strongh, keepdims=False)
mode_strongh = stats.mode(strongh)
IQR_strongh = stats.iqr(strongh)


# Calculate outliers from IQR
outliers_magnit = []
calculateOutliers(magnit, median_magnit, IQR_magnit, outliers_magnit)
    
outliers_strongh = []
calculateOutliers(strongh, median_strongh, IQR_strongh, outliers_strongh)


# Test for normal distribution
P_value_magnit = stats.normaltest(magnit)
P_value_strongh = stats.normaltest(strongh)


# Calculate coëfficient of variance
CoefficientOfVariance_magnit = stats.variation(magnit)
CoefficientOfVariance_strongh = stats.variation(strongh)


# Test if sample means differ significantly
diff_mean = stats.ttest_ind(magnit, strongh)


# Create plot figure
fig, ([ax, ax1], [ax2, ax3], [ax4, ax5]) = plt.subplots(nrows=3, ncols=2)

# Make boxplots, with 1.5 IQD outliers included
ax.boxplot(magnit)

ax.set_xlabel('Magnit B.V.')
ax.set_ylabel('Magnetisch fluxdichtheid [T]')

ax1.boxplot(strongh)

ax1.set_xlabel('Strongh B.V.')
ax1.set_ylabel('Magnetic flixdensity [T]')


# Create scatter plots for measurement data VS measurementIndex
ax2.scatter(index_magnit, magnit)

ax2.set_xlabel('Meting Magnit B.V.')
ax2.set_ylabel('Magnetisch fluxdichtheid [T]')


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


# Format plot area
plt.suptitle('Statisische plots Magnit B.V. & Strongh B.V. meet data')
fig.tight_layout()
fig = plt.gcf()
fig.set_size_inches(18, 9, forward=True)