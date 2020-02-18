import numpy as np
import pandas as pd
import statsmodels.api as sm

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


# Example for a simulated sample

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

# Variance Function

v_mu = VarFunc()

mut = g_mu.inverse(np.matmul(X, betat))
phit = g_phi.inverse(np.matmul(Z, thetat))

# Generate the response variables

y = np.random.beta(a=mut*phit, b=(1-mut)*phit, size=n)
y = y.reshape((n, 1))

df = pd.DataFrame({'resp': y[:, 0], 'cov': x1[:, 0]})

# Obtaining initial guesses

guess_coef = sm.GLM(y, X, family=Binomial()).fit(disp=False).params

mu_hat = g_mu.inverse(np.dot(X, guess_coef)).reshape((n, 1))

sigma_2_hat = np.sum((y-mu_hat)**2/(v_mu(mu_hat)))/((n-2))

phi_hat = 1/(sigma_2_hat) - 1

theta_0 = np.array([g_phi(phi_hat)])

theta_start = np.concatenate([theta_0, np.zeros(q-1)])

# Obtaining the EM estimates

# Tolerance level
tol = 10**(-8)

# Initial tolerance

tolerance = 1

count = 0

betaold = guess_coef
thetaold = theta_start


while(tolerance > tol):

    # Loglikelihood function
    def loglik(x):
        beta = x[0:p]
        theta = x[p:(p+q)]
        mu = g_mu.inverse(np.dot(X, beta)).reshape((n, 1))
        phi = g_phi.inverse(np.dot(Z, theta)).reshape((n, 1))
        phiold = g_phi.inverse(np.dot(Z, thetaold)).reshape((n, 1))
        return -np.sum(phi*(mu*np.log(y/(1-y))+digamma(phiold)+np.log(1-y))-gammaln(mu*phi) -
                       gammaln((1-mu)*phi)-np.log(y*(1-y))-digamma(phiold)-np.log(1-y)-phiold)

    # Gradient function
    def grad(x):
        beta = x[0:p]
        theta = x[p:(p+q)]
        mu = g_mu.inverse(np.dot(X, beta)).reshape((n, 1))
        phi = g_phi.inverse(np.dot(Z, theta)).reshape((n, 1))
        phiold = g_phi.inverse(np.dot(Z, thetaold)).reshape((n, 1))
        grad1 = np.matmul(
            X.T, (np.log(y/(1-y))-digamma(mu*phi)+digamma((1-mu)*phi))*mu*(1-mu)*phi)
        grad2 = np.matmul(Z.T, (mu*np.log(y/(1-y))+digamma(phiold) +
                                np.log(1-y)-mu*digamma(mu*phi)-(1-mu)*digamma((1-mu)*phi))*phi)
        grad = np.zeros_like(x)
        grad[0:p] = -grad1[:, 0]
        grad[p:(p+q)] = -grad2[:, 0]
        return grad

    # Minimize minus likelihood for the M step
    param_start = np.concatenate([betaold, thetaold])
    fit = minimize(loglik, param_start,  jac=grad, method='BFGS')
    param_new = fit.x
    betaold = param_new[0:p]
    thetaold = param_new[p:(p+q)]

    tolerance = np.abs(loglik(param_new) - loglik(param_start))
    count += 1

print('Number of iterations: ', count)

print('Estimates (Beta):', param_new[0:p], '\n',
      'Estimates (Theta):', param_new[p:(p+q)])


# Computing the expected Fisher's matrix to obtain standard errors


def FisherInf(x):
    betaFISHER = x[0:p]
    thetaFISHER = x[p:(p+q)]
    muF = g_mu.inverse(np.dot(X, betaFISHER)).reshape((n, 1))
    phiF = g_phi.inverse(np.dot(Z, thetaFISHER)).reshape((n, 1))
    yast = g_mu(y).reshape((n, 1))
    dmudeta = g_mu.inverse_deriv(g_mu(muF)).reshape((n, 1))
    d2mudeta = g_mu.inverse_deriv2(g_mu(muF)).reshape((n, 1))
    dphideta = g_phi.inverse_deriv(g_phi(phiF)).reshape((n, 1))
    d2phideta = g_phi.inverse_deriv2(g_phi(phiF)).reshape((n, 1))
    muast = digamma(muF*phiF) - digamma((1-muF)*phiF).reshape((n, 1))
    v = muF*(yast-muast)+digamma(phiF)-digamma((1-muF)*phiF)+np.log(1-y)
    v = v.reshape((n, 1))
    a = polygamma(1, (1-muF)*phiF)+polygamma(1, muF*phiF)
    a = a.reshape((n, 1))
    b = polygamma(1, (1-muF)*phiF)*(1-muF)**2 + \
        polygamma(1, muF*phiF)*muF**2-polygamma(1, phiF)
    b = b.reshape((n, 1))
    Qbb = np.diagflat(phiF*(phiF*a*dmudeta**2-(yast-muast)*d2mudeta))
    Qbt = np.diagflat(
        ((phiF*(muF*a-polygamma(1, (1-muF)*phiF))+muast-yast)*dmudeta*dphideta))
    Qtt = np.diagflat((b*dphideta**2-v*d2phideta))
    Q = np.concatenate([np.concatenate([Qbb, Qbt], axis=1),
                        np.concatenate([Qbt, Qtt], axis=1)])
    W = np.concatenate([np.concatenate([X, 0*Z], axis=1),
                        np.concatenate([0*X, Z], axis=1)])
    return(np.matmul(np.matmul(W.T, Q), W))

# Standard errors for the MLEs
SEMV = np.sqrt(np.diag(np.linalg.inv(FisherInf(param_new))))

print('Standard Errors for Beta:',SEMV[0:p],'\n','Standard Errors for Theta:',SEMV[p:(p+q)])

print('Z values for Beta:',param_new[0:p]/SEMV[0:p],'\n','Z values for Theta:',param_new[p:(p+q)]/SEMV[p:(p+q)])

print('p-values for Beta:',1-norm.cdf(abs(param_new[0:p]/SEMV[0:p])/2),'\n','p-values for Theta:',1-norm.cdf(abs(param_new[p:(p+q)]/SEMV[p:(p+q)])/2))




# Observed Fisher's Information Matrix

def ObsFisherInf(x):
    betaOF = x[0:p]
    thetaOF = x[p:(p+q)]
    muOF = g_mu.inverse(np.dot(X, betaOF)).reshape((n, 1))
    phiOF = g_phi.inverse(np.dot(Z, thetaOF)).reshape((n, 1))
    auxG1 = (phiOF**2)*(polygamma(1,muOF*phiOF)+polygamma(1,(1-muOF)*phiOF))*(muOF**2)*((1-muOF)**2) - phiOF*(np.log(y/(1-y))-digamma(muOF*phiOF)+digamma((1-muOF)*phiOF))*muOF*(1-muOF)*(1-2*muOF)
    G1 = np.diagflat(auxG1)

    auxG2 = (phiOF**2)*((muOF**2)*polygamma(1,muOF*phiOF)+((1-muOF)**2)*polygamma(1,(1-muOF)*phiOF))-phiOF*(muOF*np.log(y/(1-y))+digamma(phiOF)+np.log(1-y)-muOF*digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))
    G2 = np.diagflat(auxG2)

    auxG3 = phiOF*muOF*(1-muOF)*(-np.log(y/(1-y))+digamma(muOF*phiOF)-digamma((1-muOF)*phiOF)+phiOF*muOF*polygamma(1,muOF*phiOF)-phiOF*(1-muOF)*polygamma(1,(1-muOF)*phiOF))
    G3 =     G2 = np.diagflat(auxG3)

    D2QBeta = np.matmul(np.matmul(X.T,G1),X)
    D2QBA= np.matmul(np.matmul(X.T,G3),Z)
    D2QBeta = np.concatenate([D2QBeta,D2QBA],axis=1)
    D2QAlpha = np.matmul(np.matmul(Z.T,G2),Z)
    D2QAlpha = np.concatenate([D2QBA.T,D2QAlpha],axis=1)
    D2Q = np.concatenate([D2QBeta,D2QAlpha]) 

    grad1 = np.matmul(X.T,((np.log(y/(1-y))-digamma(muOF*phiOF)+digamma((1-muOF)*phiOF))*muOF*(1-muOF)*phiOF))
    grad2 = np.matmul(Z.T,((muOF*np.log(y/(1-y))+digamma(phiOF)+np.log(1-y)-muOF*digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))*phiOF))

    auxG4 = (phiOF**2)*(polygamma(1,phiOF)+(digamma(phiOF)+np.log(1-y)+muOF*np.log(y/(1-y))-muOF*digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))**2)
    auxG4TEMP = phiOF*(muOF*np.log(y/(1-y))+digamma(phiOF)+np.log(1-y)-muOF*digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))
    G4 = np.diagflat(auxG4)
    G4TEMP = np.matmul(auxG4TEMP,auxG4TEMP.T)
    G4TEMP = G4TEMP - np.diagflat(np.diag(G4TEMP))
    G4 = G4+G4TEMP
    #G4 = G4TEMP + diag(c(psigamma(phi,1)*phi^2))

    DQ2Beta = np.matmul(grad1,grad1.T)
    DQBetaAlfa = np.matmul(grad1,grad2.T)
    DQ2Alfa = np.matmul(np.matmul(Z.T,G4),Z)

    DQ21 = np.concatenate([DQ2Beta,DQBetaAlfa],axis=1)
    DQ22 = np.concatenate([DQBetaAlfa.T,DQ2Alfa], axis=1)
    DQ2 = np.concatenate([DQ21,DQ22])

    ######### FISHER INFORMATION MATRIX

    INF = D2Q-DQ2
    return INF

# Standard Errors obtained from the Observed information matrix

if np.sum(np.diag(np.linalg.inv(ObsFisherInf(param_new)))>0) < p+q:
    print('Error: Unable to compute, matrix not positive definite')
else:
    SEMV_Obs = np.sqrt(np.diag(np.linalg.inv(ObsFisherInf(param_new))))

    print('Standard Errors for Beta (Observed):',SEMV_Obs[0:p],'\n','Standard Errors for Theta (Observed):',SEMV_Obs[p:(p+q)])

    print('Z values for Beta (Observed):',param_new[0:p]/SEMV_Obs[0:p],'\n','Z values for Theta (Observed):',param_new[p:(p+q)]/SEMV_Obs[p:(p+q)])

    print('p-values for Beta (Observed):',1-norm.cdf(abs(param_new[0:p]/SEMV_Obs[0:p])/2),'\n','p-values for Theta (Observed):',1-norm.cdf(abs(param_new[p:(p+q)]/SEMV_Obs[p:(p+q)])/2))


