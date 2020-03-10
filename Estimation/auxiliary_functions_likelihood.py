import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize



from auxiliary_functions_estimation import (
    correct_dimension,
    estimate_mean,
    estimate_precision,
)


def loglikelihood_function_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision,
):
    """
    Obtain the log-likelihood function for the beta regression model.
    
    For more details we refer to:
    Ferrari, S. L. P., Cribari-Neto, F. (2004). Beta regression
    for modeling rates and proportions. J. Appl. Statist. 31, 799–815
    
    Simas, A. B., Barreto-Souza, W., Rocha, A. V. (2010). Improved
    estimators for a general class of beta regression models. Computational
    Statistics and Data Analysis, 54, 348–366
    
    We return the value with the minus sign since we want to maximize
    the log-likelihood function and we will utilize the minimize function to
    optimizate.
    
    :param endog (array_like): 1d array of endogenous response variable.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.

    :param exog_precision (array_like): A nobs x q array where nobs is the 
    number of observations and q is the number of precision regressors. 
    An intercept is not included by default and should be added by the user.

    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit' and that the default precision link 
    is None.

    :param param_mean: 1d array of mean regression parameters.
    
    :param param_precision: 1d array of precision regression parameters.
    
    """

    estimated_mean = estimate_mean(exog_mean, param_mean, bounded_reg_link)
    estimated_precision = estimate_precision(
        exog_precision, param_precision, bounded_reg_link
    )

    loglik = -np.sum(
        gammaln(estimated_precision)
        - gammaln(estimated_mean * estimated_precision)
        - gammaln((1 - estimated_mean) * estimated_precision)
        + (estimated_mean * estimated_precision -1) * np.log(endog)
        + ((1 - estimated_mean) * estimated_precision - 1) * np.log(1 - endog)
    )  
    return loglik


def score_mean_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision,
):
    """
    Computes the score vector with respect to the mean regression parameters. 
    
    For more details we refer to:
    Ferrari, S. L. P., Cribari-Neto, F. (2004). Beta regression
    for modeling rates and proportions. J. Appl. Statist. 31, 799–815
    
    Simas, A. B., Barreto-Souza, W., Rocha, A. V. (2010). Improved
    estimators for a general class of beta regression models. Computational
    Statistics and Data Analysis, 54, 348–366
    
    :param endog (array_like): 1d array of endogenous response variable.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.
    
    :param exog_precision (array_like): A nobs x q array where nobs is the 
    number of observations and q is the number of precision regressors. 
    An intercept is not included by default and should be added by the user.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit' and that the default precision link 
    is None.

    :param param_mean: 1d array of mean regression parameters.   
    
    :param param_precision: 1d array of precision regression parameters.
    """
    estimated_mean = estimate_mean(exog_mean, param_mean, bounded_reg_link)
    estimated_precision = estimate_precision(
        exog_precision, param_precision, bounded_reg_link
    )
    
    score_mean = np.matmul(
        exog_mean.T,
        (
            np.log(endog)  - np.log(1 - endog)
            - digamma(estimated_mean * estimated_precision)
            + digamma((1 - estimated_mean) * estimated_precision)
        )
        * bounded_reg_link.dmudeta(estimated_mean)
        * estimated_precision,
    )

    return correct_dimension(score_mean)


def score_precision_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision,
):
    """
    Computes the score vector with respect to the precision regression 
    parameters. 
    
    For more details we refer to:
    Ferrari, S. L. P., Cribari-Neto, F. (2004). Beta regression
    for modeling rates and proportions. J. Appl. Statist. 31, 799–815
    
    Simas, A. B., Barreto-Souza, W., Rocha, A. V. (2010). Improved
    estimators for a general class of beta regression models. Computational
    Statistics and Data Analysis, 54, 348–366
    
    :param endog (array_like): 1d array of endogenous response variable.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.
    
    :param exog_precision (array_like): A nobs x q array where nobs is the 
    number of observations and q is the number of precision regressors. 
    An intercept is not included by default and should be added by the user.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit' and that the default precision link 
    is None.

    :param param_mean: 1d array of mean regression parameters.   
    
    :param param_precision: 1d array of precision regression parameters.
    """
    estimated_mean = estimate_mean(exog_mean, param_mean, bounded_reg_link)
  
    estimated_precision = estimate_precision(
        exog_precision, param_precision, bounded_reg_link
    )   
    
    if exog_precision is None:
        exog_precision = param_precision * np.ones_like(estimated_mean)
    
    score_precision = np.matmul(
        exog_precision.T,
        (
            estimated_mean * (
            np.log(endog) - np.log(1 - endog)
            - digamma(estimated_mean * estimated_precision)
            + digamma((1 - estimated_mean) * estimated_precision)
            )
            + digamma(estimated_precision) 
            - digamma((1 - estimated_mean) * estimated_precision)
            + np.log(1 - endog)
        )
        * bounded_reg_link.dphideta(estimated_precision),
    )

    return correct_dimension(score_precision)


def score_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision
):
    """
    Return minus score vector of the beta regression model.
    
    For more details we refer to:
    Ferrari, S. L. P., Cribari-Neto, F. (2004). Beta regression
    for modeling rates and proportions. J. Appl. Statist. 31, 799–815
    
    Simas, A. B., Barreto-Souza, W., Rocha, A. V. (2010). Improved
    estimators for a general class of beta regression models. Computational
    Statistics and Data Analysis, 54, 348–366
    
    :param endog (array_like): 1d array of endogenous response variable.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.
    
    :param exog_precision (array_like): A nobs x q array where nobs is the 
    number of observations and q is the number of precision regressors. 
    An intercept is not included by default and should be added by the user.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit' and that the default precision link 
    is None.

    :param param_mean: 1d array of mean regression parameters.   
    
    :param param_precision: 1d array of precision regression parameters.
    
    :param previous_precision: 1d array of the regression parameters 
    related to the precision in the previous EM-step.
    """
    score_mean = score_mean_beta(
        endog,
        exog_mean,
        exog_precision,
        bounded_reg_link,
        param_mean,
        param_precision,
    )

    score_precision = score_precision_beta(
        endog,
        exog_mean,
        exog_precision,
        bounded_reg_link,
        param_mean,
        param_precision,
    )

    score = np.concatenate([-score_mean, -score_precision])
    return np.ndarray.flatten(score)

def maximize_loglikelihood_beta(
    param_mean_start,
    param_precision_start,
    endog,
    exog_mean,
    bounded_reg_link,
    method,
    exog_precision=None,
    **kwargs,
):
    """
    Maximize the loglikelihood function. 
    
    For more details we refer to:
    Ferrari, S. L. P., Cribari-Neto, F. (2004). Beta regression
    for modeling rates and proportions. J. Appl. Statist. 31, 799–815
    
    Simas, A. B., Barreto-Souza, W., Rocha, A. V. (2010). Improved
    estimators for a general class of beta regression models. Computational
    Statistics and Data Analysis, 54, 348–366
    
    :param param_mean_start (array_like): 1d array of initial guesses for
    the mean regression parameters.
    
    :param param_precision_start (array_like): 1d array of initial guesses
    for the precision regression parameters.
    
    :param endog (array_like): 1d array of endogenous response variable.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.
    
    :param exog_precision (array_like): A nobs x q array where nobs is the 
    number of observations and q is the number of precision regressors. 
    An intercept is not included by default and should be added by the user.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit' and that the default precision link 
    is None.

    :param em_optim_params (dict): A dictionary of parameters related
    to optimization:
        em_tolerance: the error tolerance for convergence in the EM procedure.
        max_em_iterations: the maximum number of iterations in the EM 
                           procedure.
        
    **kwargs: additional parameters to be passed to the minimize function
                  from the scipy.optimize module.
    """
    k = exog_mean.shape[1]
    if exog_precision is None:
        q = 1
    else:
        q = exog_precision.shape[1]
    param_start = np.concatenate([param_mean_start, param_precision_start])
    fit = minimize(
        lambda x: loglikelihood_function_beta(
            endog = endog,
            exog_mean = exog_mean,
            exog_precision=exog_precision,
            bounded_reg_link=bounded_reg_link,
            param_mean=x[:k],
            param_precision=x[k : (k + q)],
        ),
        param_start,
        jac=lambda x: score_beta(
            endog = endog,
            exog_mean = exog_mean,
            exog_precision=exog_precision,
            bounded_reg_link=bounded_reg_link,
            param_mean=x[:k],
            param_precision=x[k : (k + q)],
        ),
        method=method,
        **kwargs,
    )
    return fit