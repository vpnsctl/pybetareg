from patsy import dmatrices, dmatrix

import auxiliary_functions_estimation as afe
import auxiliary_functions_likelihood as afl
from boundedreglink import BoundedRegLink


def betareg_fit_likelihood(
    endog,
    exog_mean,
    exog_precision=None,
    bounded_reg_link=BoundedRegLink(),
    method="BFGS",
    initial_guess_method="Default",
    **kwargs
):
    """
    Function to fit a beta regression model with dispersion covariates 
    with parameters estimated via EM algorithm.
    
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

    :param method (str): the method to be used in the optimization. See
    scipy.optmize for a complete list of available methods.
    
    :param initial_guess_method (str): The method to be used to obtain the 
    initial guesses. The options are: 'Default' (estimate the mean by 
    quasi-likelihood) and 'R' (use the same strategy used in R's version).
        
    **kwargs: additional parameters to be passed to the minimize function
                  from the scipy.optimize module.
                  
    The tolerance parameter is calculated as described in the above reference.
    """
    reg_params = {
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

    param_mean_start = afe.initial_guess_mean(
        endog, exog_mean, bounded_reg_link, initial_guess_method
    )
    param_precision_start = afe.initial_guess_precision_beta(
        endog,
        exog_mean,
        exog_precision,
        param_mean_start,
        bounded_reg_link,
        initial_guess_method,
    )

    fitted_parameters = afl.maximize_loglikelihood_beta(
        **reg_params,
        param_mean_start=param_mean_start,
        param_precision_start=param_precision_start,
        method=method,
        **kwargs
    )

    count = fitted_parameters.nit
    
    max_likelihood = -fitted_parameters.fun

    estimated_parameters = fitted_parameters.x

    return {
        "estimated_mean_parameters": estimated_parameters[:k],
        "estimated_precision_parameters": estimated_parameters[k : (k + q)],
        "exog_mean": exog_mean,
        "exog_precision": exog_precision,
        "endog": endog,
        "numerical_iterations": count,
        "link_mean": bounded_reg_link.get_link_mean_name(),
        "link_precision": bounded_reg_link.get_link_precision_name(),
        "link_functions": bounded_reg_link,
        "Log-Likelihood": max_likelihood,
    }


def betareg_formula_likelihood(
    formula,
    data,
    bounded_reg_link=BoundedRegLink(),
    method="BFGS",
    initial_guess_method="Default",
    **kwargs
):
    """
    R-Style version of betareg, using formulas.
    
    :param formula (str): A string containing the formula to be used. The
    format of the formula is the following:
        endog ~ mean_var1 + ... + mean_varn | prec_var1 + ... + prec_varm
    where endog is the column name of the endogenous variable (response
    variable) to be considered, mean_var1, ..., mean_varn are the column
    names of the variables to be considered for the mean regression
    or combination of the names using the operators *, /, **, :, 
    for further details on these operators, visit the patsy documentation:
        https://patsy.readthedocs.io/en/latest/formulas.html#operators
    Finally, prec_var1, ..., prec_varn are the column names to be considered
    for the precision regression or a combination of the names using the
    operators *, /, **, : .
    
    * If there is a regression for the precision parameter and no link function
    for the precision parameter is given, we set the default link as
    'log'. 
    
    * If there is no regression for the precision parameter, then the
    link function for the precision parameter is the identity.
    
    :param data (pandas.DataFrame): A pandas.DataFrame consisting of the
    data to be used in the regression.
    
    :param bounded_reg_link: An instance of BoundedRegLink. Recall that
    the default mean link is 'logit' and that the default precision link 
    is None.

    :param method (str): the method to be used in the optimization. See
    scipy.optmize for a complete list of available methods.
    
    :param initial_guess_method (str): The method to be used to obtain the 
    initial guesses. The options are: 'Default' (estimate the mean by 
    quasi-likelihood) and 'R' (use the same strategy used in R's version).
        
    **kwargs: additional parameters to be passed to the minimize function
                  from the scipy.optimize module.
    
    """
    _bounded_reg_link = bounded_reg_link
    formula_tmp = formula.split(sep="|")
    if len(formula_tmp) == 1:
        matrix_tmp = dmatrices(formula, data)
        endog = dmatrices(matrix_tmp)[0]
        exog_mean = dmatrices(matrix_tmp)[1]
        exog_precision = None
        _mean_link = bounded_reg_link.get_link_mean_name()
        _bounded_reg_link = BoundedRegLink(
            link_mean=_mean_link, link_precision="identity"
        )
        del matrix_tmp
    elif len(formula_tmp) == 2:
        formula_tmp2 = "~" + formula_tmp[1]
        matrix_tmp_mean = dmatrices(formula_tmp[0], data)
        matrix_tmp_precision = dmatrix(formula_tmp2, data)
        endog = matrix_tmp_mean[0]
        exog_mean = matrix_tmp_mean[1]
        exog_precision = matrix_tmp_precision
        if bounded_reg_link.get_link_precision_name() == "None":
            _mean_link = bounded_reg_link.get_link_mean_name()
            _bounded_reg_link = BoundedRegLink(
                link_mean=_mean_link, link_precision="log"
            )
        del matrix_tmp_mean, matrix_tmp_precision
    else:
        raise ValueError("Could not parse the formula correctly")

    fit = betareg_fit_likelihood(
        endog,
        exog_mean,
        exog_precision,
        _bounded_reg_link,
        method,
        initial_guess_method,
        **kwargs
    )
    return fit
