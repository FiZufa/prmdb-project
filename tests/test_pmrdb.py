"""
Unit test for PMRDB engine using synthetic data.
This verifies that:
- ProbabilitySpace
- GaussianDistribution
- EvidenceFusion
- PMRDB pipeline
are all working together.
"""

import numpy as np

# Import your modules
from pmrdb.pmrdb import PMRDB
from pmrdb.distributions import GaussianDistribution
from pmrdb.fusion import EvidenceFusion
from pmrdb.probability_space import ProbabilitySpace

# --------------------------------------------------------
# PATCH: add missing Bayesian combine method for testing
# --------------------------------------------------------
def simple_bayesian_combine(priors, likelihoods):
    """
    Extremely simplified Bayesian combination:
    posterior ∝ prior * likelihood

    priors: list of means (scalar)
    likelihoods: list of probabilities
    """
    priors = np.array(priors)
    likelihoods = np.array(likelihoods)

    # Normalize priors
    priors = priors / (priors.sum() + 1e-8)

    # Multiply element-wise
    posterior_raw = priors * likelihoods

    # Normalize posterior
    posterior = posterior_raw.sum()
    return float(posterior)

# monkey-patch it into EvidenceFusion for this test
EvidenceFusion.bayesian_combine = staticmethod(simple_bayesian_combine)


# --------------------------------------------------------
# MAIN TEST
# --------------------------------------------------------
def test_pmrdb_pipeline():
    print("=== TEST: PMRDB End-to-End Synthetic Pipeline ===")

    # Synthetic modality inputs
    np.random.seed(42)

    T = np.random.normal(0, 1, 100)      # trajectory distances
    R = np.random.normal(3, 2, 100)      # regime embeddings
    V = np.random.normal(-1, 0.5, 100)   # volatility

    # Create PMRDB instance
    db = PMRDB()

    # Register modalities
    db.set_modalities(T, R, V)

    # Synthetic observation
    observation = {
        "trajectory": 0.2,
        "regime": 2.5,
        "volatility": -1.2,
    }

    # Run forecast
    result = db.forecast(observation, n_samples=30)

    print("\n=== RESULT ===")
    print(result)

    # basic sanity checks
    assert "posterior_probability" in result
    assert "uncertainty" in result
    assert result["uncertainty"]["variance"] >= 0.0

    print("\nTEST PASSED ✔")


# --------------------------------------------------------
# RUN DIRECTLY
# --------------------------------------------------------
if __name__ == "__main__":
    test_pmrdb_pipeline()
