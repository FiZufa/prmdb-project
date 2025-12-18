# test_distributions.py

import numpy as np
from pmrdb.distributions import (
    GaussianDistribution,
    BetaDistribution,
    DirichletDistribution
)

def test_gaussian():
    print("=== Testing GaussianDistribution ===")

    dist = GaussianDistribution(mean=5.0, var=4.0)  # std = 2.0

    # Test sampling
    samples = dist.sample(10000)
    print("Sample mean:", np.mean(samples))
    print("Sample var :", np.var(samples))

    # Test PDF at known positions
    print("PDF(5) =", dist.pdf(5))
    print("PDF(0) =", dist.pdf(0))

    print()


def test_beta():
    print("=== Testing BetaDistribution ===")

    dist = BetaDistribution(a=2.0, b=5.0)

    samples = dist.sample(5000)
    print("Sample mean:", np.mean(samples))
    print("Sample var :", np.var(samples))

    # Check PDF consistency
    print("PDF(0.5) =", dist.pdf(0.5))
    print("PDF(0.1) =", dist.pdf(0.1))

    print()


def test_dirichlet():
    print("=== Testing DirichletDistribution ===")

    dist = DirichletDistribution([2, 3, 4])  # 3-dimensional

    samples = dist.sample(5000)
    sample_mean = np.mean(samples, axis=0)

    print("Sample mean vector:", sample_mean)
    print("Row sums (should be 1):", np.mean(samples.sum(axis=1)))

    # Check PDF value
    x = np.array([0.2, 0.3, 0.5])
    print("PDF([0.2, 0.3, 0.5]) =", dist.pdf(x))

    print()


if __name__ == "__main__":
    test_gaussian()
    test_beta()
    test_dirichlet()
