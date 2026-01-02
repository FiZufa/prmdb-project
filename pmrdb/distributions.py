# pmrdb/distributions.py

import numpy as np
from scipy.stats import norm, beta, dirichlet
# import networkx as nx
import torch
import math

class GaussianDistribution:
    def __init__(self, mean, var):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.var = torch.as_tensor(var, dtype=torch.float32)
        self.std = torch.sqrt(self.var)

        self.dist = torch.distributions.Normal(self.mean, self.std)

    def pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.exp(self.dist.log_prob(x))

    def log_pdf(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.dist.log_prob(x)

    def sample(self, n=1):
        return self.dist.sample((n,))

    @classmethod
    def from_pdf(cls, pdf_fn):
        raise NotImplementedError(
            "General non-parametric pdf construction not implemented."
        )


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
