# pmrdb/pmrdb.py

import numpy as np
from pmrdb.probability_space import ProbabilitySpace
from pmrdb.distributions import GaussianDistribution
from pmrdb.fusion import EvidenceFusion


class PMRDB:

    def __init__(self):
        self.space = ProbabilitySpace()
        self.fusion = EvidenceFusion()

        # Default binary forecasting
        self.space.set_priors({"UP": 0.5, "DOWN": 0.5})

    # -----------------------------------------------------------
    # 1. Register distributions for modalities
    # -----------------------------------------------------------
    def set_modalities(self, T: np.ndarray, R: np.ndarray, V: np.ndarray):

        # Build Gaussian distributions
        T_dist = GaussianDistribution(T.mean(), T.var() + 1e-6)
        R_dist = GaussianDistribution(R.mean(), R.var() + 1e-6)
        V_dist = GaussianDistribution(V.mean(), V.var() + 1e-6)

        # Register likelihood functions
        self.space.register_likelihood(
            "T",
            lambda x, h: T_dist.pdf(x)
        )

        self.space.register_likelihood(
            "R",
            lambda x, h: R_dist.pdf(x)
        )

        self.space.register_likelihood(
            "V",
            lambda x, h: V_dist.pdf(x)
        )

    # -----------------------------------------------------------
    # 2. Compute posterior given evidence
    # -----------------------------------------------------------
    def compute_posterior(self, obs):
        """
        obs = {"T": value, "R": value, "V": value}
        """
        return self.space.posterior(obs)

    # -----------------------------------------------------------
    # 3. Monte Carlo uncertainty
    # -----------------------------------------------------------
    def estimate_uncertainty(self, posterior_samples):
        arr = np.array(posterior_samples)

        return {
            "mean": float(arr.mean()),
            "variance": float(arr.var()),
            "aleatoric": float(arr.var() * 0.5),
            "epistemic": float(arr.var() * 0.5),
            "total": float(arr.var())
        }

    # -----------------------------------------------------------
    # 4. Full pipeline
    # -----------------------------------------------------------
    def forecast(self, observation, n_samples=50):
        samples = []

        for _ in range(n_samples):
            post = self.compute_posterior(observation)["UP"]
            samples.append(post)

        return {
            "posterior_probability_UP": float(np.mean(samples)),
            "uncertainty": self.estimate_uncertainty(samples)
        }
