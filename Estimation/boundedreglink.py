from numpy import exp

# Link functions for the mean regression
from statsmodels.genmod.families.links import logit, cauchy, cloglog, probit

# Link functions for the precision regression
from statsmodels.genmod.families.links import (
    log,
    inverse_squared,
    inverse_power,
    identity,
)


class BoundedRegLink:
    """
    A class containing the link functions for the mean and precision
    parameters of a regression with response variable having bounded
    support.
    
    :param link_mean: A string containing the link function for the 
    mean regression. The current options are 'logit', 'cauchy', 'cloglog' 
    and 'probit'. The standard link is 'logit'.
    
    :param link_precision: A string containing the link function for the 
    precision regression. The current options are 'log', 'inverse_squared' 
    and 'inverse_power'.
    
    We initialize the link_precision with None since most regressions do not
    have a regression structure for the precision parameter.
    
    For further references see:
    Simas, Barreto-Souza & Rocha (2010) Improved estimators for a general 
    class of beta regression models. Computational Statistics & Data Analysis,
    Volume 54, Pages 348-366
    
    Barreto-Souza & Simas (2017) Improving estimation for beta regression 
    models via em-algorithm and related diagnostic tools, Volume 87, Pages
    2847-2867
    
    Most methods are obtained from their counterparts in statsmodels module
    """

    def __init__(self, link_mean="logit", link_precision=None):
        map_link_mean = {
            "logit": logit,
            "cauchy": cauchy,
            "cloglog": cloglog,
            "probit": probit,
        }

        map_link_precision = {
            "log": log,
            "inverse_squared": inverse_squared,
            "inverse_power": inverse_power,
            "identity": identity,
        }
        self._link_mean_name = link_mean
        self._link_precision_name = link_precision
        self._link_mean = map_link_mean[link_mean]
        self._link_mean_instance = self._link_mean()
        if not (link_precision is None):
            self._link_precision = map_link_precision[link_precision]
            self._link_precision_instance = self._link_precision()
        else:
            self._link_precision_name = "None"

    def link_mean(self, p):
        """Returns the value of the mean link function evaluated at p"""
        return self._link_mean_instance(p)

    def get_link_mean_name(self):
        return self._link_mean_name

    def get_link_precision_name(self):
        return self._link_precision_name
    
    def get_link_mean(self):
        return self._link_mean_instance
    
    def get_link_precision(self):
        return self._link_precision_instance

    def deriv_mean(self, p):
        """
        Returns the value of the derivative of the mean link function 
        evaluated at p
        """
        return self._link_mean_instance.deriv(p)

    def deriv2_mean(self, p):
        """
        Returns the value of the second derivative of the mean link function 
        evaluated at p
        """
        return self._link_mean_instance.deriv2(p)

    def inverse_mean(self, z):
        """
        Returns the value of the inverse of the mean link function evaluated
        at z.
        """
        if self.get_link_mean_name() == "logit":
            inv_function = 1 / (1.0 + exp(-z))
        else:
            inv_function = self._link_mean_instance.inverse(z)

        return inv_function

    def inverse_deriv_mean(self, z):
        """
        Returns the value of the derivative of the inverse of the mean link 
        function evaluated at z.
        """
        return self._link_mean_instance.inverse_deriv(z)

    def inverse_deriv2_mean(self, z):
        """
        Returns the value of the inverse of the derivative of the inverse of 
        the mean link function evaluated at z.
        """
        return self._link_mean_instance.inverse_deriv2(z)

    def link_precision(self, p):
        """Returns the value of the precision link function evaluated at p"""
        return self._link_precision_instance(p)

    def deriv_precision(self, p):
        """
        Returns the value of the derivative of the precision link function 
        evaluated at p
        """
        return self._link_precision_instance.deriv(p)

    def deriv2_precision(self, p):
        """
        Returns the value of the second derivative of the precision link 
        function evaluated at p
        """
        return self._link_precision_instance.deriv2(p)

    def inverse_precision(self, z):
        """
        Returns the value of the inverse of the precision link function 
        evaluated at z.
        """
        return self._link_precision_instance.inverse(z)

    def inverse_deriv_precision(self, z):
        """
        Returns the value of the derivative of the inverse of the precision 
        link function evaluated at z.
        """
        return self._link_precision_instance.inverse_deriv(z)

    def inverse_deriv2_precision(self, z):
        """
        Returns the value of the inverse of the derivative of the inverse of 
        the precision link function evaluated at z.
        """
        return self._link_precision_instance.inverse_deriv2(z)

    def dmudeta(self, p):
        """
        Returns the derivative of the mean with respect to the linear
        predictor evaluated at p.
        """
        return self.inverse_deriv_mean(self.link_mean(p))

    def d2mudeta(self, p):
        """
        Returns the second derivative of the mean with respect to the linear
        predictor evaluated at p.
        """
        return self.inverse_deriv2_mean(self.link_mean(p))

    def dphideta(self, p):
        """
        Returns the derivative of the precision parameter with respect to 
        the linear predictor evaluated at p.
        """
        return self.inverse_deriv_precision(self.link_precision(p))

    def d2phideta(self, p):
        """
        Returns the second derivative of the precision parameter with respect
        to the linear predictor evaluated at p.
        """
        return self.inverse_deriv2_precision(self.link_precision(p))
