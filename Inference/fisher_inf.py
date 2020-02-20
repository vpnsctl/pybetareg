import numpy as np
from scipy.special import digamma, polygamma

def getAuxiliaryValuesBeta(model):
    '''
    Auxiliary Function to return the array containing the mean, the 
    precision parameter of the model, $y^\ast = g_\mu(y)$, 
    $\mu^ast = \psi(\mu\phi) - psi((1-\mu)\phi)$, $a$, $v$ and $b$.
    
    $\psi$ being the digamma function.
    
    model: A fitted EM-Beta model.
    
    For detailed expressions of $a$, $v$ and $b$ and further details: 
    Simas AB, Barreto-Souza W, Rocha AV (2010). “Improved Estimators 
    for a General Class of Beta Regression Models.” 
    Computational Statistics & Data Analysis, 54(2), 348–366.
    '''
    est = model['est']
    p = model['p']
    q = model['q']
    link_mean = model['link_mean']
    link_phi = model['link_phi']
    X = model['X']
    Z = model['Z']
    n = model['n']
    y = model['y']
    beta = est[0:p]
    theta = est[p:(p+q)]
    g_mu = link_mean()
    g_phi = link_phi()
    mu = g_mu.inverse(np.dot(X, beta)).reshape((n, 1))
    phi = g_phi.inverse(np.dot(Z, theta)).reshape((n, 1))
    yast = g_mu(y).reshape((n, 1))
    muast = digamma(mu*phi) - digamma((1-mu)*phi).reshape((n, 1))
    a = polygamma(1, (1-mu)*phi)+polygamma(1, mu*phi)
    a = a.reshape((n, 1))
    v = mu*(yast-muast)+digamma(phi)-digamma((1-mu)*phi)+np.log(1-y)
    v = v.reshape((n, 1))
    b = polygamma(1, (1-mu)*phi)*(1-mu)**2 + \
        polygamma(1, mu*phi)*mu**2-polygamma(1, phi)
    b = b.reshape((n, 1))    
    return {'mean': mu, 'precision': phi, 'y_ast': yast, 'mu_ast': muast,
            'a': a, 'v': v, 'b': b}

def dmudeta(model):
    '''
    Auxiliary function to compute the derivative of the mean with respect
    to the linear predictor with respect to $\mu$ ($\eta_\mu$). 
    
    We have $g_\mu(\mu_i) = \eta_{i,\mu},$ and $\eta_{i,\mu} = x_i^T \beta.$
    
    model: A fitted EM-Beta model.
    '''
    link_mean = model['link_mean']
    n = model['n']
    g_mu = link_mean()
    AuxValues = getAuxiliaryValuesBeta(model)
    mu = AuxValues['mean']
    return g_mu.inverse_deriv(g_mu(mu)).reshape((n, 1))

def d2mudeta(model):
    '''
    Auxiliary function to compute the  second derivative of the mean with 
    respect to the linear predictor with respect to $\mu$ ($\eta_\mu$). 
    
    We have $g_\mu(\mu_i) = \eta_{i,\mu},$ and $\eta_{i,\mu} = x_i^T \beta.$
    
    model: A fitted EM-Beta model.
    '''
    link_mean = model['link_mean']
    n = model['n']
    g_mu = link_mean()
    AuxValues = getAuxiliaryValuesBeta(model)
    mu = AuxValues['mean']
    return g_mu.inverse_deriv2(g_mu(mu)).reshape((n, 1))

def dphideta(model):
    '''
    Auxiliary function to compute the derivative of the precision parameter
    with respect to the linear predictor with respect to $\phi$ ($\eta_\phi$). 
    
    We have $g_\phi(\phi_i) = \eta_{i,\phi},$ and 
    $\eta_{i,\phi} = z_i^T \theta.$
    
    model: A fitted EM-Beta model.
    '''
    link_phi = model['link_phi']
    n = model['n']
    g_phi = link_phi()
    AuxValues = getAuxiliaryValuesBeta(model)
    phi = AuxValues['precision']   
    return g_phi.inverse_deriv(g_phi(phi)).reshape((n, 1))

def d2phideta(model):
    '''
    Auxiliary function to compute the second derivative of the precision 
    parameter with respect to the linear predictor with respect to 
    $\phi$ ($\eta_\phi$). 
    
    We have $g_\phi(\phi_i) = \eta_{i,\phi},$ and 
    $\eta_{i,\phi} = z_i^T \theta.$
    
    model: A fitted EM-Beta model.
    '''
    link_phi = model['link_phi']
    n = model['n']
    g_phi = link_phi()
    AuxValues = getAuxiliaryValuesBeta(model)
    phi = AuxValues['precision']            
    return g_phi.inverse_deriv2(g_phi(phi)).reshape((n, 1))
 
def FisherInf(model):
    '''
    Function to obtain the Fisher's information matrix with respect to 
    the regression parameters beta and theta.
    
    model: A fitted EM-Beta model.
    '''
    X = model['X']
    Z = model['Z']
    AuxValues = getAuxiliaryValuesBeta(model)
    mu = AuxValues['mean']
    phi = AuxValues['precision']
    yast = AuxValues['y_ast']
    muast = AuxValues['mu_ast']
    a = AuxValues['a']
    v = AuxValues['v']
    b = AuxValues['b']
    dmudeta_fit = dmudeta(model)
    d2mudeta_fit = dmudeta(model)
    dphideta_fit = dphideta(model)
    d2phideta_fit = d2phideta(model)
    Qbb = np.diagflat(phi*(phi*a*dmudeta_fit**2-(yast-muast)*d2mudeta_fit))
    Qbt = np.diagflat(
    ((phi*(mu*a-polygamma(1, (1-mu)*phi))+ \
      muast-yast)*dmudeta_fit*dphideta_fit))
    
    Qtt = np.diagflat((b*dphideta_fit**2-v*d2phideta_fit))
    Q = np.concatenate([np.concatenate([Qbb, Qbt], axis=1),
                        np.concatenate([Qbt, Qtt], axis=1)])
    W = np.concatenate([np.concatenate([X, 0*Z], axis=1),
                        np.concatenate([0*X, Z], axis=1)])
    return (np.matmul(np.matmul(W.T, Q), W))

def D2Q_Obs_Fisher(model):
    '''
    Auxiliary function to compute the observed Fisher's information matrix
    
    model: A fitted EM-Beta model.
    '''
    X = model['X']
    Z = model['Z']
    y = model['y']
    AuxValues = getAuxiliaryValuesBeta(model)
    mu = AuxValues['mean']
    phi = AuxValues['precision']
    dmudeta_fit = dmudeta(model)
    d2mudeta_fit = dmudeta(model)
    dphideta_fit = dphideta(model)
    d2phideta_fit = d2phideta(model)
    
    auxG1 = (phi**2)*(polygamma(1, mu*phi)+\
             polygamma(1, (1-mu)*phi))*(dmudeta_fit**2) - \
             phi*(np.log(y/(1-y))-digamma(mu*phi) +
             digamma((1-mu)*phi))*d2mudeta_fit
             
    G1 = np.diagflat(auxG1)

    auxG2 = (dphideta_fit**2)*((mu**2)*polygamma(1, mu*phi)+ \
             ((1-mu)**2)*polygamma(1, (1-mu)*phi))-d2phideta_fit * \
             (mu*np.log(y/(1-y))+digamma(phi)+np.log(1-y)-mu *
             digamma(mu*phi)-(1-mu)*digamma((1-mu)*phi))
             
    G2 = np.diagflat(auxG2)

    auxG3 = dphideta_fit*dmudeta_fit*(-np.log(y/(1-y))+digamma(mu*phi)- \
                        digamma((1-mu)*phi) + \
                        phi*mu*polygamma(1, mu*phi)- \
                        phi*(1-mu)*polygamma(1, (1-mu)*phi))
    
    G3 = np.diagflat(auxG3)

    D2QBeta = np.matmul(np.matmul(X.T, G1), X)
    D2QBA = np.matmul(np.matmul(X.T, G3), Z)
    D2QBeta = np.concatenate([D2QBeta, D2QBA], axis=1)
    D2QAlpha = np.matmul(np.matmul(Z.T, G2), Z)
    D2QAlpha = np.concatenate([D2QBA.T, D2QAlpha], axis=1)
    D2Q = np.concatenate([D2QBeta, D2QAlpha])
    
    return D2Q

def DQ2_Obs_Fisher(model):
    '''
    Auxiliary function to compute the observed Fisher's information matrix
    
    model: A fitted EM-Beta model.
    '''
    X = model['X']
    Z = model['Z']
    y = model['y']
    AuxValues = getAuxiliaryValuesBeta(model)
    mu = AuxValues['mean']
    phi = AuxValues['precision']
    dmudeta_fit = dmudeta(model)
    dphideta_fit = dphideta(model)  
    
    grad1 = np.matmul(X.T, ((np.log(y/(1-y))-digamma(mu*phi) + \
                             digamma((1-mu)*phi))*dmudeta_fit*phi))
    
    grad2 = np.matmul(Z.T, ((mu*np.log(y/(1-y))+digamma(phi) + \
                            np.log(1-y)-mu*digamma(mu*phi) - \
                            (1-mu)*digamma((1-mu)*phi))*dphideta_fit))

    
    auxG4 = (dphideta_fit**2)*(polygamma(1, phi)+(digamma(phi)+np.log(1-y)+ \
                        mu*np.log(y/(1-y))-mu*digamma(mu*phi)- \
                        (1-mu)*digamma((1-mu)*phi))**2)
    
    auxG4TEMP = dphideta_fit*(mu*np.log(y/(1-y))+digamma(phi)+ \
                       np.log(1-y) - mu*digamma(mu*phi) - \
                       (1-mu)*digamma((1-mu)*phi))
    
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
    return DQ2    

def ObsFisherInf(model):
    '''
    Function to obtain the observed Fisher's information matrix 
    with respect to the regression parameters beta and theta.
    
    model: A fitted EM-Beta model.
    '''    
    return D2Q_Obs_Fisher(model) - DQ2_Obs_Fisher(model)
