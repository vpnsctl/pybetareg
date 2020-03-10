import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.varfuncs import Binomial as VarFunc

FLOAT_EPS = np.finfo(float).eps


def correct_dimension(data):
    """
    Turn a zero-dimensional array into a one-dimensional array.
    
    :param data (array-like):
    """
    new_data = data
    if new_data.ndim == 0:
        new_data.shape += (1, 1)
    elif new_data.ndim == 1:
        new_data.shape += (1,)
    return new_data


def estimate_mean(exog_mean, param_mean, bounded_reg_link):
    """
    A function that obtains an estimate of the mean from a given set 
    of (mean) regression parameters.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.
    
    :param param_mean (array_like): 1d array of mean regression 
    parameters.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit'.
    """

    estimated_mean = bounded_reg_link.inverse_mean(
        np.dot(exog_mean, param_mean)
    )

    estimated_mean = np.clip(estimated_mean, FLOAT_EPS, 1.0 - FLOAT_EPS)

    return correct_dimension(estimated_mean)


def estimate_precision(exog_precision, param_precision, bounded_reg_link):
    """
    A function that obtains an estimate of the precision from a given set 
    of (precision) regression parameters.
    
    :param exog_precision (array_like): A nobs x q array where nobs is the 
    number of observations and q is the number of precision regressors. 
    An intercept is not included by default and should be added by the user.
    
    :param param_precision (array_like): 1d array of precision regression 
    parameters.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default precision link is None.
    """

    if exog_precision is None:
        estimated_precision = param_precision
    else:
        estimated_precision = bounded_reg_link.inverse_precision(
            np.dot(exog_precision, param_precision)
        )

    return correct_dimension(estimated_precision)


def initial_guess_mean(endog, exog_mean, bounded_reg_link, method="Default"):
    """
    A function that obtains an initial guess for the regression 
    parameters related to the mean in a bounded data regression model. The
    initial guess is obtained from a quasi-likelihood regression.
    
    :param endog (array_like): 1d array of endogenous response variable.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default precision link is None.
    
    :param method (str): The method to be used to obtain the initial guesses.
    The options are: 'Default' (estimate the mean by quasi-likelihood) and 
    'R' (use the same strategy used in R's version).
    """
    if method == "Default":
        initial_guess_mean_param = (
            sm.GLM(
                endog,
                exog_mean,
                family=Binomial(link=bounded_reg_link.get_link_mean()),
            )
            .fit(disp=False)
            .params
        )
    elif method == "R":
        endog_mod = bounded_reg_link.link_mean(endog)
        initial_guess_mean_param = (
            sm.OLS(endog_mod, exog_mean).fit(disp=False).params
        )
    else:
        raise ValueError(
            "Please enter a valid method for the initial guess, "+
            "the options are 'Default' and 'R'."
        )

    return correct_dimension(initial_guess_mean_param)


def initial_guess_precision_beta(
    endog,
    exog_mean,
    exog_precision,
    initial_guess_mean_param,
    bounded_reg_link,
    method="Default",
):
    """   
    A function that obtains an initial guess for the parameters related
    to the precision parameter in a beta regression model. The initial 
    guess is given by a moment-based estimator.
    
    :param endog (array_like): 1d array of endogenous response variable.
    
    :param exog_mean (array_like): A nobs x k array where nobs is the number 
    of observations and k is the number of mean regressors. An intercept is 
    not included by default and should be added by the user.
    
    :param initial_guess_mean_param (dict): A result from the
    function initial_guess_mean.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default precision link is None.
    """
    k = exog_mean.shape[1]
    nobs = len(endog)

    if method == "Default":
        estimated_mean = estimate_mean(
            exog_mean, initial_guess_mean_param, bounded_reg_link
        )
        variance_function = VarFunc()

        estimated_dispersion = np.sum(
            (endog - estimated_mean) ** 2 / (variance_function(estimated_mean))
        )
        estimated_precision = (nobs - k - 2) / estimated_dispersion
    elif method == "R":
        endog_mod = bounded_reg_link.link_mean(endog)
        model_ols = sm.OLS(endog_mod, exog_mean).fit(disp=False)
        fitted = model_ols.predict()
        fitted = correct_dimension(fitted)
        res = endog_mod - fitted
        mu_fitted = bounded_reg_link.inverse_mean(fitted)
        dlink = bounded_reg_link.deriv_mean(mu_fitted)
        sigma2 = np.sum(res ** 2 / ((nobs - k) * dlink ** 2))
        estimated_precision = np.mean(mu_fitted * (1 - mu_fitted) / sigma2) - 1
    else:
        raise ValueError(
            "Please enter a valid method for the initial guess, "+
            "the options are 'Default' and 'R'."
        )

    if exog_precision is None:
        estimated_prec_param = estimated_precision
    else:
        q = exog_precision.shape[1]
        estimated_prec_param = np.hstack(
            [
                bounded_reg_link.link_precision(estimated_precision),
                np.zeros(q - 1),
            ]
        )

    return correct_dimension(np.array(estimated_prec_param))
