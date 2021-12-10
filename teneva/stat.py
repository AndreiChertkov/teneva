import numpy as np


def confidence(F, alpha=.05):
    """Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the eCDF.

    The arguments are: the empirical distributions F (array_like) and alpha for a (1 - alpha) % confidence band. It is based on the DKW inequality. See
    "Wasserman, L. 2006. `All of Nonparametric Statistics`. Springer" for more
    details.

    """
    eps = np.sqrt(np.log(2./alpha) / (2 * len(F)))
    return np.clip(F - eps, 0, 1), np.clip(F + eps, 0, 1)


def get_cdf(x):
    _x = np.array(x, copy=True)
    _x.sort()
    _y = np.linspace(1./len(_x), 1, len(_x))

    _x = np.r_[-np.inf, _x]
    _y = np.r_[0, _y]

    def cdf(z):
        return _y[np.searchsorted(_x, z, 'right') - 1]

    return cdf
