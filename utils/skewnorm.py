from scipy.stats import skewnorm
import numpy as np

def get_cdf(u, components, w):
    cdf = np.zeros_like(u)
    for i in range(len(components)):
        cdf += w[i] * skewnorm.cdf(u,*components[i])
    return cdf

def get_empirical_cdf(u):
    nu = len(u)
    empirical_cdf = np.linspace(0,1,nu) + (1/nu)
    return empirical_cdf

def yang_dist(x,y,p=2):
    x = np.array(x)
    y = np.array(y)
    gt = x >= y
    dP = ((x[gt] - y[gt]).sum()**p + (y[~gt] - x[~gt]).sum()**p) ** (1/p)
    dPn = dP / sum([max(abs(xi),abs(yi),abs(xi-yi)) for xi,yi in zip(x,y)])
    return dPn

def get_cdf_dist(x, components, w):
    x = np.array(x)
    u = sorted(np.unique(x[np.random.randint(0,len(x),size=len(x))]))
    empirical_cdf = get_empirical_cdf(u)
    model_cdf = get_cdf(u, components, w)
    return yang_dist(empirical_cdf,model_cdf)
    
def joint_densities(x, params, weights):
    """
    weighted pdfs of a mixture of skew normal distributions
    """
    return np.array([w * skewnorm.pdf(x, a, loc, scale) for (a, loc, scale), w in zip(params, weights)])