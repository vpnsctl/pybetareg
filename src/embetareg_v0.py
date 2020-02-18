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

# Algorithm to fit beta regression with EM-algorithm


def fitBetaEM(X: 'array or df', y: 'array or df', Z: 'array or df',
              link_mean: 'link mean', link_phi: 'link phi',
              tol: 'float', maxit=5000, showit=True):
    p = X.shape[1]
    q = Z.shape[1]
    n = X.shape[0]
    g_mu = link_mean()
    g_phi = link_phi()
    v_mu = VarFunc()
    guess_coef = sm.GLM(y, X, family=Binomial()).fit(disp=False).params
    mu_hat = g_mu.inverse(np.dot(X, guess_coef)).reshape((n, 1))
    sigma_2_hat = np.sum((y-mu_hat)**2/(v_mu(mu_hat)))/((n-2))
    phi_hat = 1/(sigma_2_hat) - 1
    theta_0 = np.array([g_phi(phi_hat)])
    theta_start = np.concatenate([theta_0, np.zeros(q-1)])

    tolerance = 1
    count = 0

    betaold = guess_coef
    thetaold = theta_start

    while(tolerance > tol and count < maxit):

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

        tolerance = np.maximum(np.abs(
            loglik(param_new) - loglik(param_start)), np.linalg.norm(param_new-param_start))
        count += 1
        if(count == maxit):
            print(
                'Warning: Algorithm did not converge within the maximum number of iterations')
    if showit == True:
        print('Number of iterations: ', count)
    return {'est': param_new, 'p': p, 'q': q, 'link_mean': link_mean,
            'link_phi': link_phi, 'X': X, 'Z': Z, 'y': y, 'n': n}

# Expected Fisher Information Matrix


def FisherInf(x):
    est = x['est']
    p = x['p']
    q = x['q']
    link_mean = x['link_mean']
    link_phi = x['link_phi']
    X = x['X']
    Z = x['Z']
    n = x['n']
    y = x['y']
    g_mu = link_mean()
    g_phi = link_phi()
    betaFISHER = est[0:p]
    thetaFISHER = est[p:(p+q)]
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


# Observed Fisher's Information Matrix

def ObsFisherInf(x):
    est = x['est']
    p = x['p']
    q = x['q']
    link_mean = x['link_mean']
    link_phi = x['link_phi']
    X = x['X']
    Z = x['Z']
    n = x['n']
    y = x['y']
    g_mu = link_mean()
    g_phi = link_phi()
    betaOF = est[0:p]
    thetaOF = est[p:(p+q)]
    muOF = g_mu.inverse(np.dot(X, betaOF)).reshape((n, 1))
    phiOF = g_phi.inverse(np.dot(Z, thetaOF)).reshape((n, 1))
    auxG1 = (phiOF**2)*(polygamma(1, muOF*phiOF)+polygamma(1, (1-muOF)*phiOF))*(muOF**2)*((1-muOF)**2) - \
        phiOF*(np.log(y/(1-y))-digamma(muOF*phiOF) +
               digamma((1-muOF)*phiOF))*muOF*(1-muOF)*(1-2*muOF)
    G1 = np.diagflat(auxG1)

    auxG2 = (phiOF**2)*((muOF**2)*polygamma(1, muOF*phiOF)+((1-muOF)**2)*polygamma(1, (1-muOF)*phiOF))-phiOF * \
        (muOF*np.log(y/(1-y))+digamma(phiOF)+np.log(1-y)-muOF *
         digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))
    G2 = np.diagflat(auxG2)

    auxG3 = phiOF*muOF*(1-muOF)*(-np.log(y/(1-y))+digamma(muOF*phiOF)-digamma((1-muOF)*phiOF) +
                                 phiOF*muOF*polygamma(1, muOF*phiOF)-phiOF*(1-muOF)*polygamma(1, (1-muOF)*phiOF))
    G3 = np.diagflat(auxG3)

    D2QBeta = np.matmul(np.matmul(X.T, G1), X)
    D2QBA = np.matmul(np.matmul(X.T, G3), Z)
    D2QBeta = np.concatenate([D2QBeta, D2QBA], axis=1)
    D2QAlpha = np.matmul(np.matmul(Z.T, G2), Z)
    D2QAlpha = np.concatenate([D2QBA.T, D2QAlpha], axis=1)
    D2Q = np.concatenate([D2QBeta, D2QAlpha])

    grad1 = np.matmul(X.T, ((np.log(y/(1-y))-digamma(muOF*phiOF) +
                             digamma((1-muOF)*phiOF))*muOF*(1-muOF)*phiOF))
    grad2 = np.matmul(Z.T, ((muOF*np.log(y/(1-y))+digamma(phiOF)+np.log(1-y) -
                             muOF*digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))*phiOF))

    auxG4 = (phiOF**2)*(polygamma(1, phiOF)+(digamma(phiOF)+np.log(1-y)+muOF *
                                             np.log(y/(1-y))-muOF*digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))**2)
    auxG4TEMP = phiOF*(muOF*np.log(y/(1-y))+digamma(phiOF)+np.log(1-y) -
                       muOF*digamma(muOF*phiOF)-(1-muOF)*digamma((1-muOF)*phiOF))
    G4 = np.diagflat(auxG4)
    G4TEMP = np.matmul(auxG4TEMP, auxG4TEMP.T)
    G4TEMP = G4TEMP - np.diagflat(np.diag(G4TEMP))
    G4 = G4+G4TEMP

    DQ2Beta = np.matmul(grad1, grad1.T)
    DQBetaAlfa = np.matmul(grad1, grad2.T)
    DQ2Alfa = np.matmul(np.matmul(Z.T, G4), Z)

    DQ21 = np.concatenate([DQ2Beta, DQBetaAlfa], axis=1)
    DQ22 = np.concatenate([DQBetaAlfa.T, DQ2Alfa], axis=1)
    DQ2 = np.concatenate([DQ21, DQ22])

    # FISHER INFORMATION MATRIX

    INF = D2Q-DQ2
    return INF


def summary(x, Cov='Obs'):
    est = x['est']
    p = x['p']
    q = x['q']
    if(Cov == 'Obs'):
        SEMV = np.sqrt(np.diag(np.linalg.inv(ObsFisherInf(x))))
    else:
        SEMV = np.sqrt(np.diag(np.linalg.inv(FisherInf(x))))
    for i in range(p+q):
        print('Estimate: ', est[i], '\t Std. Error: ', SEMV[i], '\t Z value: ',
              est[i]/SEMV[i], '\t p-value: ', 1-norm.cdf(abs(est[i]/SEMV[i])/2))


def residual(x, type='Score', plot='index', envelope=False, envelope_num=100):
    est = x['est']
    y = x['y']
    X = x['X']
    Z = x['Z']
    p = x['p']
    q = x['q']
    n = x['n']
    beta_hat = est[0:p]
    theta_hat = est[p:(p+q)]
    link_mean = x['link_mean']
    link_phi = x['link_phi']
    g_mu = link_mean()
    g_phi = link_phi()
    v_mu = VarFunc()
    mu_hat = g_mu.inverse(np.dot(X, beta_hat)).reshape((n, 1))
    phi_hat = g_phi.inverse(np.dot(Z, theta_hat)).reshape((n, 1))
    y_ast = np.log(y/(1-y))
    mu_ast = digamma(mu_hat*phi_hat) - digamma((1-mu_hat)*phi_hat)
    a_hat = polygamma(1, mu_hat*phi_hat) + polygamma(1, (1-mu_hat)*phi_hat)
    if type == 'Score':
        res = (y_ast - mu_ast)/np.sqrt(a_hat)
    elif type == 'Pearson':
        res = (np.sqrt(1+phi_hat))*(y - mu_hat)/(np.sqrt(v_mu(mu_hat)))

    if plot == 'index':
        df = pd.DataFrame(data=res, columns=['Residuals'])
        plt.plot('index', 'Residuals', data=df.reset_index(), marker = '.', linestyle = 'none')
    elif plot == 'quantile':
        sorted_res = np.sort(res, axis=0)
        sorted_res = sorted_res.reshape((n, 1))
        sep = np.linspace(0.01, 0.99, num=n)
        qnorm = norm.ppf(sep)
        qnorm = qnorm.reshape((n, 1))
        res_q = np.concatenate((sorted_res, qnorm), axis=1)
        df = pd.DataFrame(data=res_q, columns=[
                          'Residuals', 'Normal Quantiles'])
        #sns.scatterplot(x = 'Normal Quantiles', y = 'Residuals', data = df)

        if envelope == True:
            res_matrix = np.zeros(shape=(n, envelope_num))
            for i in range(envelope_num):
                ynew = np.random.beta(
                    a=mu_hat*phi_hat, b=(1-mu_hat)*phi_hat, size=n)
                ynew = ynew.reshape((n, 1))
                x_res = fitBetaEM(X, ynew, Z, link_mean,
                                  link_phi, 10**(-3), showit=False)
                est_res = x_res['est']
                beta_new = est_res[0:p]
                theta_new = est_res[p:(p+q)]
                mu_new = g_mu.inverse(np.dot(X, beta_new)).reshape((n, 1))
                phi_new = g_phi.inverse(np.dot(Z, theta_new)).reshape((n, 1))
                y_ast_new = np.log(ynew/(1-ynew))
                mu_ast_new = digamma(mu_new*phi_new) - \
                    digamma((1-mu_new)*phi_new)
                a_hat_new = polygamma(1, mu_new*phi_new) + \
                    polygamma(1, (1-mu_new)*phi_new)
                if type == 'Score':
                    res_new = (y_ast_new - mu_ast_new)/np.sqrt(a_hat_new)
                    res_new.sort(axis=0)
                    res_new = res_new.reshape((n,))
                    res_matrix[:, i] = res_new
                elif type == 'Pearson':
                    res_new = (np.sqrt(1+phi_new)) * \
                        (ynew - mu_new)/(np.sqrt(v_mu(mu_new)))
                    res_new.sort(axis=0)
                    res_new = res_new.reshape((n,))
                    res_matrix[:, i] = res_new
            res_matrix.sort(axis=1)
            res_low = (res_matrix[:, 2] + res_matrix[:, 3])/2
            res_up = (res_matrix[:, (envelope_num-3)] +
                      res_matrix[:, (envelope_num - 2)])/2
            res_low = res_low.reshape(n, 1)
            res_env_low = np.concatenate((res_low, qnorm), axis=1)
            res_up = res_up.reshape(n, 1)
            res_env_up = np.concatenate((res_up, qnorm), axis=1)
            df_low = pd.DataFrame(data=res_env_low, columns=[
                                  'lower bound', 'Normal Quantiles'])
            df_up = pd.DataFrame(data=res_env_up, columns=[
                                 'upper bound', 'Normal Quantiles'])
            plt.plot('Normal Quantiles', 'lower bound',
                     data=df_low, color='blue')
            plt.plot('Normal Quantiles', 'upper bound',
                     data=df_up, color='blue')
            df_merge = pd.merge(left=df_low, right=df_up,
                                on='Normal Quantiles')
            plt.fill_between(x='Normal Quantiles', y1='lower bound',
                             y2='upper bound', alpha=0.2, data=df_merge)
        plt.plot('Normal Quantiles', 'Residuals', data=df,
                 marker='.', linestyle='none', color='black')
        plt.show()
    return res
