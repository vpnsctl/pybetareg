import pandas as pd

from likelihood_estimation import betareg_formula_likelihood
from em_estimation import betareg_formula_em

GasolineYield = pd.read_csv("Datasets/GasolineYield.csv")
FoodExpenditure = pd.read_csv("Datasets/foodexpenditure.csv")
Methylation = pd.read_csv("Datasets/methylation-test.csv")

gy_logit = betareg_formula_likelihood(
    "gas_yield ~ C(batch, Treatment(10)) + temp", GasolineYield
)

gy_logit_nm = betareg_formula_likelihood(
    "gas_yield ~ C(batch, Treatment(10)) + temp", GasolineYield, 
    method = "Nelder-Mead"
)

gy_logit_em = betareg_formula_em(
    "gas_yield ~ C(batch, Treatment(10)) + temp", GasolineYield
)

gy_logit_R = betareg_formula_likelihood(
    "gas_yield ~ C(batch, Treatment(10)) + temp", GasolineYield,
    initial_guess_method="R"
)


gy_logit_R_nm = betareg_formula_likelihood(
    "gas_yield ~ C(batch, Treatment(10)) + temp", GasolineYield,
    initial_guess_method="R", method = "Nelder-Mead"
)

gy_logit_R_em = betareg_formula_em(
    "gas_yield ~ C(batch, Treatment(10)) + temp", GasolineYield,
    initial_guess_method="R"
)

gy_logit_em_maxit = betareg_formula_em(
    "gas_yield ~ C(batch, Treatment(10)) + temp", GasolineYield,
    em_optim_params={"max_em_iterations": 5000, "em_tolerance": 10 ** (-6),
                             "method": "Nelder-Mead",}
)

gy_logit2 = betareg_formula_likelihood(
    "gas_yield ~ C(batch, Treatment(10)) + temp | temp", GasolineYield
)


fe_beta = betareg_formula_likelihood(
    "I(food/income) ~ income + persons", data=FoodExpenditure
)

fe_beta2 = betareg_formula_likelihood(
    "I(food/income) ~ income + persons | persons", data=FoodExpenditure
)

gy_loglog = betareg_formula_likelihood(
    "gas_yield ~ C(batch, Treatment(1)) + temp",
    data=GasolineYield,
    bounded_reg_link=BoundedRegLink(link_mean="cloglog"),
)

ml_logit = betareg_formula_likelihood(
    "methylation ~ gender + CpG | age", data=Methylation
)
