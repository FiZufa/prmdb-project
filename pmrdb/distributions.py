# pmrdb/distributions.py

import numpy as np
from scipy.stats import norm, beta, dirichlet

class GaussianDistribution:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.std = np.sqrt(var)

    def pdf(self, x):
        return norm.pdf(x, self.mean, self.std)

    def sample(self, n=1):
        return np.random.normal(self.mean, self.std, n)

    @classmethod
    def from_pdf(cls, pdf_fn):
        raise NotImplementedError("General non-parametric pdf construction not implemented.")


class BetaDistribution:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        return beta.pdf(x, self.a, self.b)

    def sample(self, n=1):
        return np.random.beta(self.a, self.b, n)


class DirichletDistribution:
    def __init__(self, alpha_vec):
        self.alpha = np.array(alpha_vec)

    def pdf(self, x):
        return dirichlet.pdf(x, self.alpha)

    def sample(self, n=1):
        return dirichlet.rvs(self.alpha, size=n)
