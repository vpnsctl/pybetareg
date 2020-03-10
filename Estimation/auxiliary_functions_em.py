import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize

from auxiliary_functions_estimation import (
    correct_dimension,
    estimate_mean,
    estimate_precision,
)


def q_function_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision,
    previous_precision,
):
    """
    Obtain the Q function for the EM estimation of the beta regression.
    In the iterative procedure, we have to carry the previous value of 
    the precision parameter.
    
    For more details we refer to:
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
    We return the value with the minus sign since we want to maximize
    the Q function and we will utilize the minimize function to
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
    
    :param previous_precision: 1d array of the regression parameters 
    related to the precision in the previous EM-step.
    """

    estimated_mean = estimate_mean(exog_mean, param_mean, bounded_reg_link)
    estimated_precision = estimate_precision(
        exog_precision, param_precision, bounded_reg_link
    )

    return -np.sum(
        estimated_precision
        * (
            estimated_mean * np.log(endog / (1 - endog))
            + digamma(previous_precision)
            + np.log(1 - endog)
        )
        - gammaln(estimated_mean * estimated_precision)
        - gammaln((1 - estimated_mean) * estimated_precision)
        - np.log(endog * (1 - endog))
        - digamma(previous_precision)
        - np.log(1 - endog)
        - previous_precision
    )


def grad_q_mean_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision,
):
    """
    Computes the gradient of the Q function with respect to the mean
    regression parameters. 
    
    For more details we refer to:
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
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

    grad_mean = np.matmul(
        exog_mean.T,
        (
            np.log(endog / (1 - endog))
            - digamma(estimated_mean * estimated_precision)
            + digamma((1 - estimated_mean) * estimated_precision)
        )
        * bounded_reg_link.dmudeta(estimated_mean)
        * estimated_precision,
    )

    return correct_dimension(grad_mean)


def grad_q_precision_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision,
    previous_precision,
):
    """
    Computes the gradient of the Q function with respect to the precision
    regression parameters. 
    
    For more details we refer to:
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
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
    estimated_mean = estimate_mean(exog_mean, param_mean, bounded_reg_link)
    estimated_precision = estimate_precision(
        exog_precision, param_precision, bounded_reg_link
    )
    if exog_precision is None:
        exog_precision = param_precision * np.ones_like(estimated_mean)
        
    grad_precision = np.matmul(
        exog_precision.T,
        (
            estimated_mean * np.log(endog / (1 - endog))
            + digamma(previous_precision)
            + np.log(1 - endog)
            - estimated_mean * digamma(estimated_mean * estimated_precision)
            - (1 - estimated_mean)
            * digamma((1 - estimated_mean) * estimated_precision)
        )
        * bounded_reg_link.dphideta(estimated_precision),
    )

    return correct_dimension(grad_precision)


def grad_q_beta(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean,
    param_precision,
    previous_precision,
):
    """
    Return minus the gradient of the Q function
    
    For more details we refer to:
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
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
    grad_mean = grad_q_mean_beta(
        endog,
        exog_mean,
        exog_precision,
        bounded_reg_link,
        param_mean,
        param_precision,
    )

    grad_precision = grad_q_precision_beta(
        endog,
        exog_mean,
        exog_precision,
        bounded_reg_link,
        param_mean,
        param_precision,
        previous_precision,
    )

    grad = np.concatenate([-grad_mean, -grad_precision])          
    return np.ndarray.flatten(grad)


def previous_precision_update(
    previous_precision_param, exog_precision, bounded_reg_link
):
    """
    Takes the precision regression parameters as input and returns
    the estimated precision. This function is needed in the E-step
    of the EM algorithm.
    
    :param previous_precision_param (array_like): 1d array of the
    regression parameters related to the precision during the EM algorithm.
    
    :param exog_precision (array_like): A nobs x q array where nobs is the 
    number of observations and q is the number of precision regressors. 
    An intercept is not included by default and should be added by the user.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit' and that the default precision link 
    is None. 
    """
    if exog_precision is None:
        previous_precision = previous_precision_param
    else:
        previous_precision = bounded_reg_link.inverse_precision(
            np.dot(exog_precision, previous_precision_param)
        )
    return previous_precision


def maximize_q_function_beta(
    param_start,
    previous_precision,
    endog,
    exog_mean,
    bounded_reg_link,
    method="BFGS",
    exog_precision=None,
    **kwargs,
):
    """
    Minimize the minus Q function. This is the M-step in the EM-algorithm.
    
    For more details we refer to:
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
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
    fit = minimize(
        lambda x: q_function_beta(
            endog,
            exog_mean,
            exog_precision,
            bounded_reg_link,
            x[:k],
            x[k : (k + q)],
            previous_precision,
        ),
        param_start,
        jac=lambda x: grad_q_beta(
            endog,
            exog_mean,
            exog_precision,
            bounded_reg_link,
            x[:k],
            x[k : (k + q)],
            previous_precision,
        ),
        method=method,
        **kwargs,
    )
    return fit


def compute_tolerance_beta(
    param_new,
    param_start,
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    previous_precision,
):
    """
    Compute the tolerance as given in:
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
    :param param_new: The most recent estimated parameters.
    
    :param param_start: the previous estimated parameters.
    
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
    
    :param previous_precision: 1d array of the regression parameters 
    related to the precision in the previous EM-step.
    """
    k = exog_mean.shape[1]
    if exog_precision is None:
        q = 1
    else:
        q = exog_precision.shape[1]
    tolerance = np.maximum(
        np.abs(
            q_function_beta(
                endog,
                exog_mean,
                exog_precision,
                bounded_reg_link,
                param_new[:k],
                param_new[k : (k + q)],
                previous_precision,
            )
            - q_function_beta(
                endog,
                exog_mean,
                exog_precision,
                bounded_reg_link,
                param_start[:k],
                param_start[k : (k + q)],
                previous_precision,
            )
        ),
        np.linalg.norm(param_new - param_start),
    )
    return tolerance


def em_loop(
    endog,
    exog_mean,
    exog_precision,
    bounded_reg_link,
    param_mean_start,
    param_precision_start,
    em_optim_params={
        "em_tolerance": 10 ** (-6),
        "max_em_iterations": 5000,
        "method": "BFGS",
    },
    **kwargs):
    '''
    Runs the EM procedure until convergence (or when reaching the 
    maximum number of iterations).
    
    For more details we refer to:
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
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

    :param param_mean_start (array_like): initial guesses of the mean 
    regression parameters.
    
    :param param_precision_start (array_like): initial guess of the 
    precision regression parameters.

    :param optim_params (dict): A dictionary of parameters related
    to optimization:
        em_tolerance: the error tolerance for convergence in the EM procedure.
        max_em_iterations: the maximum number of iterations in the EM 
                           procedure.
        
    **kwargs: additional parameters to be passed to the minimize function
                  from the scipy.optimize module.
                  
    The EM tolerance parameter is calculated as described in the above 
    reference.
    '''
    em_params = {
        "endog": endog,
        "exog_mean": exog_mean,
        "bounded_reg_link": bounded_reg_link,
        "exog_precision": exog_precision,
    }
    
    k = exog_mean.shape[1]
    if exog_precision is None:
        q = 1
    else:
        q = exog_precision.shape[1]
    
    em_tolerance = 1
    count = 0

    param_start = np.concatenate([param_mean_start, param_precision_start])
    previous_precision_param = param_precision_start

    while (
        em_tolerance > em_optim_params["em_tolerance"]
        and count < em_optim_params["max_em_iterations"]
    ):
        previous_precision = previous_precision_update(
            previous_precision_param, exog_precision, bounded_reg_link
        )

        fit = maximize_q_function_beta(
            **em_params,
            param_start=param_start,
            previous_precision=previous_precision,
            method=em_optim_params["method"],
            **kwargs,
        )
        param_new = fit.x
        em_tolerance = compute_tolerance_beta(
            **em_params,
            param_new=param_new,
            param_start=param_start,
            previous_precision=previous_precision,
        )
        param_start = param_new
        previous_precision_param = correct_dimension(param_new[k : (k + q)])
        count += 1
        
    return {'estimated_parameters': param_new, 'count': count}