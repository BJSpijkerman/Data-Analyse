# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:50:50 2023

@author: brams
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

magnitDat = pd.read_csv('Magnit.B.V.csv', encoding='utf-8')
stronghDat = pd.read_csv('Strongh B.V.csv', encoding='utf-8')

magnit = magnitDat['Magnetische fluxdichtheid [T]']
index_magnit = magnitDat['Meting [#]']
strongh = stronghDat['Magnetische fluxdichtheid [T]']
index_strongh = stronghDat['Meting [#]']

DescriptionMagnit = stats.describe(magnit)
DescriptionStrongh = stats.describe(strongh)

P_value_magnit = stats.normaltest(magnit)
P_value_strongh = stats.normaltest(strongh)

CoefficientOfVariance_magnit = stats.variation(magnit)
CoefficientOfVariance_strongh = stats.variation(strongh)

fig, ([ax, ax1], [ax2, ax3]) = plt.subplots(nrows=2, ncols=2)

ax.boxplot(magnit)
ax1.boxplot(strongh)

ax2.scatter(index_magnit, magnit)
ax3.scatter(index_strongh, strongh)
