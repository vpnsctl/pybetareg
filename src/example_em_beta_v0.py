from embetareg_v0 import *

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.special import gammaln

from scipy.special import digamma, polygamma

from scipy.stats import norm

from scipy.optimize import minimize

# Link functions for the mean

from statsmodels.genmod.families.links import logit, cauchy, cloglog, probit

# Link functions for the dispersion parameter

from statsmodels.genmod.families.links import log, inverse_squared, inverse_power


from statsmodels.genmod.families.varfuncs import Binomial as VarFunc

from statsmodels.genmod.families import Binomial

#Setting seed

np.random.seed(123)

# Sample size
n = 200

# Covariates
x1 = np.random.random(n).reshape((n, 1))
x2 = np.random.binomial(1, 0.3, size=n).reshape((n, 1))

X = sm.add_constant(x1)
Z = sm.add_constant(x2)

p = X.shape[1]
q = Z.shape[1]

# Theoretical parameters

betat = np.array([.5, 1])
thetat = np.array([2, 2])

# Link functions

g_mu = logit()
g_phi = log()

#Mean and precision parameters

mut = g_mu.inverse(np.matmul(X, betat))
phit = g_phi.inverse(np.matmul(Z, thetat))

# Generate the response variables

y = np.random.beta(a=mut*phit, b=(1-mut)*phit, size=n)
y = y.reshape((n, 1))

#Fit the EM beta regression model

est_BM = fitBetaEM(X, y, Z, logit, log, 10**(-8))

# Get basic summary statistics

summary(est_BM)

#Index plot of the Score residuals

score_res = residual(est_BM)

#Index plot of the Pearson residuals

pearson_res = residual(est_BM, type = 'Pearson')

#Quantile plot of score residuals with simulated envelopes

score_res = residual(est_BM, plot = 'quantile', envelope=True)

#Quantile plot of Pearson residuals with simulated envelopes

pearson_res = residual(est_BM, type = 'Pearson', plot = 'quantile', envelope = True)
