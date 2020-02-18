import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.special import gammaln

from scipy.special import digamma


from scipy.optimize import minimize

# Link functions for the mean

from statsmodels.genmod.families.links import logit, cauchy, cloglog, probit

# Link functions for the dispersion parameter

from statsmodels.genmod.families.links import log, inverse_squared, inverse_power


from statsmodels.genmod.families.varfuncs import Binomial as VarFunc

from statsmodels.genmod.families import Binomial




#Example for a simulated sample

#Sample size
n = 200

#Covariates
x1 = np.random.random(n).reshape((n,1))
x2 = np.random.binomial(1, 0.3, size = n).reshape((n,1))

X = sm.add_constant(x1)
Z = sm.add_constant(x2)

p = X.shape[1]
q = Z.shape[1]

#Theoretical parameters

betat = np.array([.5,1])
thetat = np.array([2,2])

#Link functions

g_mu = logit()
g_phi = log()

#Variance Function

v_mu = VarFunc()

mut = g_mu.inverse(np.matmul(X,betat))
phit = g_phi.inverse(np.matmul(Z,thetat))

# Generate the response variables

y = np.random.beta(a=mut*phit,b=(1-mut)*phit, size=n)
y = y.reshape((n,1))

df = pd.DataFrame({'resp':y[:,0], 'cov':x1[:,0]})

# Obtaining initial guesses

guess_coef = sm.GLM(y, X, family=Binomial()).fit(disp=False).params

mu_hat = g_mu.inverse(np.dot(X,guess_coef)).reshape((n,1))

sigma_2_hat = np.sum((y-mu_hat)**2/(v_mu(mu_hat)))/((n-2))

phi_hat = 1/(sigma_2_hat) - 1

theta_0 = np.array([g_phi(phi_hat)])

theta_start = np.concatenate([theta_0,np.zeros(q-1)])

# Obtaining the EM estimates

#Tolerance level
tol = 10**(-8)

#Initial tolerance

tolerance = 1

count = 0

betaold = guess_coef
thetaold = theta_start


while(tolerance > tol):
    
    #Loglikelihood function
    def loglik(x):
        beta = x[0:p]
        theta = x[p:(p+q)]
        mu = g_mu.inverse(np.dot(X,beta)).reshape((n,1))
        phi = g_phi.inverse(np.dot(Z,theta)).reshape((n,1))
        phiold = g_phi.inverse(np.dot(Z, thetaold)).reshape((n,1))
        return -np.sum(phi*(mu*np.log(y/(1-y))+digamma(phiold)+np.log(1-y))-gammaln(mu*phi)-
            gammaln((1-mu)*phi)-np.log(y*(1-y))-digamma(phiold)-np.log(1-y)-phiold)
 
    #Gradient function
    def grad(x):
        beta = x[0:p]
        theta = x[p:(p+q)]
        mu = g_mu.inverse(np.dot(X,beta)).reshape((n,1))
        phi = g_phi.inverse(np.dot(Z,theta)).reshape((n,1))
        phiold = g_phi.inverse(np.dot(Z, thetaold)).reshape((n,1))
        grad1 = np.matmul(X.T,(np.log(y/(1-y))-digamma(mu*phi)+digamma((1-mu)*phi))*mu*(1-mu)*phi )
        grad2 = np.matmul(Z.T,(mu*np.log(y/(1-y))+digamma(phiold)+np.log(1-y)-mu*digamma(mu*phi)-(1-mu)*digamma((1-mu)*phi))*phi)        
        grad = np.zeros_like(x)
        grad[0:p] = -grad1[:,0]
        grad[p:(p+q)] = -grad2[:,0]
        return grad
        
    #Minimize minus likelihood for the M step
    param_start = np.concatenate([betaold,thetaold])
    fit = minimize(loglik, param_start,  jac = grad,method = 'BFGS')
    param_new = fit.x
    betaold = param_new[0:p]
    thetaold = param_new[p:(p+q)]
    
    tolerance = np.abs(loglik(param_new) - loglik(param_start))
    count += 1

print('Number of iterations: ', count)
