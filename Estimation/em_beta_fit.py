import numpy as np

# Variance function of the Binomial (same for Beta):
from statsmodels.genmod.families.varfuncs import Binomial as VarFunc
# Binomial family to obtain initial guesses
from statsmodels.genmod.families import Binomial
# Statsmodel to use GLM function for the initial guesses
import statsmodels.api as sm
# Special functions needed
from scipy.special import gammaln, digamma
# Optmization function
from scipy.optimize import minimize

# Obtaining the initial guesses


def initial_guess(X, y, Z, link_mean, link_phi):
    '''
    Function to obtain initial guesses for the optmization procedure

    X: pd.DataFrame or np.array of dimension n x p. This array contains the 
    features (also known as covariates) related to the mean of the model.
    y: pd.DataFrame or np.array of dimension n x 1. This array contains the
    target variable (also known as response variable).
    Z: pd.DataFrame or np.array of dimension n x q. This array contains the
    features (also known as covariates) related to the precision parameter
    of the model.
    link_mean: An element of statsmodels.genmod.families.links. The link 
    function (also known as activation function) for the mean of the model.
    We consider the following link functions for the mean: logit, cauchy, 
    cloglog, probit.
    link_phi: An element of statsmodels.genmod.families.links. The link 
    function (also known as activation function) for the precision parameter
    of the model. We consider the following link functions for the precision
    parameter: log, inverse_squared, inverse_power.
    '''
    # n: Sample size
    n = X.shape[0]
    # q: number of covariates related to the precision parameter;
    # that is, we have
    # $g_phi(\phi_j) = \theta_1 + \theta_2 z_{2,j} + \cdots + \theta_q z_{q,j}$
    q = Z.shape[1]
    # g_mu is the link function for the mean
    g_mu = link_mean()
    # g_phi is the link function for the precision parameter
    g_phi = link_phi()
    # v_mu is the variance function of the beta distribution: V(x) = x(1-x)
    v_mu = VarFunc()
    # We obtain the initial estimates (guess_mean) for the parameters related
    # to the mean by using a quasi-likelihood estimation using the GLM function.
    guess_mean = sm.GLM(y, X, family=Binomial()).fit(disp=False).params
    # We compute the estimated values of the mean with this new parameter
    # in order to obtain initial guesses for the parameter related to the
    # precision parameter
    mu_hat = g_mu.inverse(np.dot(X, guess_mean)).reshape((n, 1))
    # We begin by estimating the variance using a moment-like estimator
    sigma_2_hat = np.sum((y-mu_hat)**2/(v_mu(mu_hat)))/((n-2))
    # Here we solve the equation $\sigma^2 = 1/(1+\phi)$ in $\phi$
    phi_hat = 1/(sigma_2_hat) - 1
    # We then obtain an initial guess for $\theta_0$ which is the
    # intercept of the regression related to the precision parameter
    theta_0 = np.array([g_phi(phi_hat)])
    # We then fill the remaining spaces with zeroes to provide
    # the initial guess (since we only estimated the intercept).
    guess_precision = np.concatenate([theta_0, np.zeros(q-1)])
    # Return vector of initial guesses
    return np.concatenate([guess_mean, guess_precision])

# Q-Function (Analogous to log-likelihood function)


def QFunction(x):
    '''
    Log-Q function to the M-step of the EM-algorithm. In this step we
    maximize this function.

    x: np.array. x consists of the vector of regression parameters 
    to be optimized. We consider x to be of the form x = [beta, theta],
    where beta is the vector of regression parameters with respect to 
    the mean and theta is the vector of regression parameters with
    respect to the precision parameter.

    This functions expects that the environment has the following variables 
    loaded:
        n: The sample size.
        p: The number of covariates in the mean regression, that is,
        such that
        $g_\mu(\mu_j)=\beta_1+\beta_2 x_{2,j}+\cdots + \beta_p x_{p,j}.$
        q: The number of covariates in the precision regression, that is,
        such that
        $g_\phi(\phi_j)=\theta_1+\theta_2 z_{2,j}+ \cdots +\theta_q z_{q,j}.$
        g_mu: The link function with respect to the mean parameter.
        g_phi: The link function with respect to the precision parameter
        X: The pd.DataFrame or np.array of dimension n x p consisting of the
        covariates for the mean parameter.
        Z: The pd.DataFrame or np.array of dimension n x q consisting of the
        covariates for the precision parameter.
        y: The pd.DataFrame or np.array of dimension n x 1 consisting of the 
        response variables.
        thetaold: np.array. An array containg the previous values of the
        theta parameters in the EM procedure (used in the M-step).

        The formula of the Q-Function:
        $Q(\theta;\theta^{(r)}) = \sum_{i=1}^n\left\{\phi_i
        \left[\mu_i\log\left(\dfrac{z_i}{1-z_i}\right)+\Psi(\phi^{(r)}_i)+
        \log(1-z_i)\right]-\log\Gamma(\mu_i\phi_i)-
        \log\Gamma((1-\mu_i)\phi_i)\right\}.$

        Here, \Psi is the digamma function and \Gamma is the gamma function.
    '''
    # Making thetaold global so this function can read it its values when
    # called by the fit function
    beta = x[0:p]
    theta = x[p:(p+q)]
    mu = g_mu.inverse(np.dot(X, beta)).reshape((n, 1))
    phi = g_phi.inverse(np.dot(Z, theta)).reshape((n, 1))

    # Notice that we are returning with the negative sign, since we want
    # to maximize and this function minimizes
    return -np.sum(phi*(mu*np.log(y/(1-y))+digamma(phiold)+np.log(1-y))
                   - gammaln(mu*phi) - gammaln((1-mu)*phi)-np.log(y*(1-y))
                   - digamma(phiold)-np.log(1-y)-phiold)


# Gradient of the Q-Function
def gradQ(x):
    '''
    Computes the gradient of the Q-Function. This function is needed in 
    the optimization procedure.

    x: np.array. x consists of the vector of regression parameters 
    to be optimized. We consider x to be of the form x = [beta, theta],
    where beta is the vector of regression parameters with respect to 
    the mean and theta is the vector of regression parameters with
    respect to the precision parameter.

    This functions expects that the environment has the following variables 
    loaded:
        n: The sample size.
        p: The number of covariates in the mean regression, that is,
        such that
        $g_\mu(\mu_j)=\beta_1+\beta_2 x_{2,j}+\cdots + \beta_p x_{p,j}.$
        q: The number of covariates in the precision regression, that is,
        such that
        $g_\phi(\phi_j)=\theta_1+\theta_2 z_{2,j}+ \cdots +\theta_q z_{q,j}.$
        g_mu: The link function with respect to the mean parameter.
        g_phi: The link function with respect to the precision parameter
        X: The pd.DataFrame or np.array of dimension n x p consisting of the
        covariates for the mean parameter.
        Z: The pd.DataFrame or np.array of dimension n x q consisting of the
        covariates for the precision parameter.
        y: The pd.DataFrame or np.array of dimension n x 1 consisting of the 
        response variables..

    '''
    # Making thetaold global so this function can read it its values when
    # called by the fit function
    beta = x[0:p]
    theta = x[p:(p+q)]
    mu = g_mu.inverse(np.dot(X, beta)).reshape((n, 1))
    phi = g_phi.inverse(np.dot(Z, theta)).reshape((n, 1))

    # grad1: vector of derivatives with respect to beta (regression parameters
    # with respect to the mean)
    grad1 = np.matmul(
        X.T, (np.log(y/(1-y))-digamma(mu*phi) +
              digamma((1-mu)*phi))*mu*(1-mu)*phi)

    # grad2: vector of derivatives with respect to theta (regression parameters
    # with respect to the precision parameter)
    grad2 = np.matmul(Z.T, (mu*np.log(y/(1-y))+digamma(phiold) +
                            np.log(1-y)-mu*digamma(mu*phi) -
                            (1-mu)*digamma((1-mu)*phi))*phi)
    # Creating a vector of the same size as x consisting of the full vector
    # of derivatives (that is, with respect to beta and with respect to theta)
    grad = np.zeros_like(x)

    # Notice that we are returning with the negative sign, since we had
    # to put such a sign in the Q-function.
    grad[0:p] = -grad1[:, 0]
    grad[p:(p+q)] = -grad2[:, 0]
    return grad


# Algorithm to fit beta regression with EM-algorithm
def fitBetaEM(X, y, Z, link_mean, link_phi, tol=10**(-6),
              maxit=5000, showit=True):
    '''
    Function to obtain the EM estimates of the parameters of the beta 
    regression model. 

    X: pd.DataFrame or np.array of dimension n x p. This array contains the 
    features (also known as covariates) related to the mean of the model.
    y: pd.DataFrame or np.array of dimension n x 1. This array contains the
    target variable (also known as response variable).
    Z: pd.DataFrame or np.array of dimension n x q. This array contains the
    features (also known as covariates) related to the precision parameter
    of the model.
    link_mean: An element of statsmodels.genmod.families.links. The link 
    function (also known as activation function) for the mean of the model.
    We consider the following link functions for the mean: logit, cauchy, 
    cloglog, probit.
    link_phi: An element of statsmodels.genmod.families.links. The link 
    function (also known as activation function) for the precision parameter
    of the model. We consider the following link functions for the precision
    parameter: log, inverse_squared, inverse_power.
    tol: Float. The maximum error tolerance for the convergence of the model 
    (further details given below).
    maxit: Integer. The maximum number of iterations. A control to avoid 
    infinite loop.
    showit: Boolean. If true, show how many iterations it took to converge. If
    it did not converge, will print maxit.

    The tol parameter is such that
    $\max(|Q(\kappa_n)-Q(\kappa_{n+1})|, \|\kappa_n - \kappa_{n+1}\|) < tol,$
    where $Q$ is the $Q$-function, $\|\cdot\|$ is the euclidean norm and
    $\kappa_n = (\beta_n,\theta_n)$ is the vector of regression parameters 
    at step $n$.
    '''
    # Making phiold global so QFunction and gradQ can read its value
    global phiold
    # p is the number of covariates related to the mean parameter
    p = X.shape[1]
    # q is the number of covariates related to the precision parameter
    q = Z.shape[1]
    # n is the sample size
    n = X.shape[0]
    # g_mu is the link function related to the mean
    g_mu = link_mean()
    # g_phi is the link function related to the precision parameter
    g_phi = link_phi()
    # v_mu is the variance function of the Beta distribution: V(x) = x(1-x)
    v_mu = VarFunc()

    # Obtaining initial guesses for the optmization procedure
    param_start = initial_guess(X, y, Z, link_mean, link_phi)
    # initiating the tolerance at one to begin the loop
    tolerance = 1
    # initiating a counter of the number of iterations
    count = 0
    # Initialization of the thetaold variable, which is needed in
    # both the Q-Function and gradQ. This variable comes from the EM
    # procedure
    thetaold = param_start[p:(p+q)]
    # Starting the loop for the EM procedure
    while(tolerance > tol and count < maxit):
        # Computing phiold, which is needed in the QFunction and gradQ
        phiold = g_phi.inverse(np.dot(Z, thetaold)).reshape((n, 1))
        # Doing the M-step:
        fit = minimize(QFunction, param_start,  jac=gradQ, method='BFGS')
        # param_new: fitted parameters
        param_new = fit.x
        # the new parameters will be the old parameters in the next iteration
        betaold = param_new[0:p]
        thetaold = param_new[p:(p+q)]
        # Updating the tolerance
        tolerance = np.maximum(np.abs(QFunction(param_new) -
                                      QFunction(param_start)),
                               np.linalg.norm(param_new-param_start))
        # Updating the param_start to the previous estimation
        param_start = np.concatenate([betaold, thetaold])
        # Updating the counter
        count += 1

    if(count == maxit):
        print('Warning: Algorithm did not converge within the \
              maximum number of iterations')
    if showit == True:
        print('Number of iterations: ', count)
    return {'est': param_new, 'p': p, 'q': q, 'link_mean': link_mean,
            'link_phi': link_phi, 'X': X, 'Z': Z, 'y': y, 'n': n}
