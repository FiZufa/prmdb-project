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
from pmrdb.pmrdb import PMRDB

def test_pmrdb_pipeline():
    print("=== TEST: PMRDB End-to-End Synthetic Pipeline ===")

    np.random.seed(42)

    # Synthetic training data
    T = np.random.normal(0, 1, 100)
    R = np.random.normal(3, 2, 100)
    V = np.random.normal(-1, 0.5, 100)

    db = PMRDB()
    db.set_modalities(T, R, V)

    # 🔧 FIX: modality names must match ("T", "R", "V")
    observation = {
        "T": 0.2,
        "R": 2.5,
        "V": -1.2,
    }

    result = db.forecast(observation, n_samples=30)

    print("\n=== RESULT ===")
    print(result)

    # 🔧 FIX: correct key name
    assert "posterior_probability_UP" in result
    assert "uncertainty" in result
    assert result["uncertainty"]["variance"] >= 0.0

    print("\nTEST PASSED ✔")

if __name__ == "__main__":
    test_pmrdb_pipeline()
