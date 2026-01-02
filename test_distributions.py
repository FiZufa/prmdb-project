# test_distributions.py

import torch
import numpy as np
from pmrdb.distributions import (
    GaussianDistribution,
    BetaDistribution,
    DirichletDistribution
)

import numpy as np
from scipy.stats import norm

def analytic_gaussian_posterior(mu0, var0, data, lik_var):
    """
    Closed-form posterior for Gaussian-Gaussian model
    """
    data = torch.as_tensor(data, dtype=torch.float32)

    n = data.numel()
    xbar = data.mean()

    var_n = 1.0 / (1.0 / var0 + n / lik_var)
    mu_n = var_n * (mu0 / var0 + n * xbar / lik_var)

    return mu_n, var_n



def test_gaussian():
    print("=== Testing GaussianDistribution (PyTorch) ===")

    torch.manual_seed(42)

    # --------------------------------------------------
    # 1. Basic distribution test
    # --------------------------------------------------
    dist = GaussianDistribution(mean=5.0, var=4.0)

    samples = dist.sample(10000)
    print("Sample mean:", samples.mean().item())
    print("Sample var :", samples.var(unbiased=False).item())

    print("PDF(5) =", dist.pdf(5.0).item())
    print("PDF(0) =", dist.pdf(0.0).item())
    print()

    # --------------------------------------------------
    # 2. Posterior verification setup
    # --------------------------------------------------
    mu0, var0 = torch.tensor(0.0), torch.tensor(1.0)
    lik_var = torch.tensor(0.25)

    true_mu = 1.5
    data = torch.normal(
        mean=true_mu,
        std=torch.sqrt(lik_var),
        size=(50,)
    )

    # --------------------------------------------------
    # 3. Analytical posterior (ground truth)
    # --------------------------------------------------
    mu_n, var_n = analytic_gaussian_posterior(
        mu0, var0, data, lik_var
    )

    print("Analytical posterior mean:", mu_n.item())
    print("Analytical posterior var :", var_n.item())
    print()

    # --------------------------------------------------
    # 4. Monte Carlo verification
    # --------------------------------------------------
    posterior = GaussianDistribution(mu_n, var_n)
    post_samples = posterior.sample(100000)

    sample_mean = post_samples.mean()
    sample_var = post_samples.var(unbiased=False)

    print("Posterior sample mean:", sample_mean.item())
    print("Posterior sample var :", sample_var.item())
    print()

    # --------------------------------------------------
    # 5. Consistency checks
    # --------------------------------------------------
    mean_error = torch.abs(sample_mean - mu_n)
    var_error = torch.abs(sample_var - var_n)

    print("Mean error:", mean_error.item())
    print("Var error :", var_error.item())

    assert mean_error < 1e-2, "Posterior mean mismatch"
    assert var_error < 1e-2, "Posterior variance mismatch"

    print("\n✅ Gaussian posterior verified successfully.\n")



# def test_gaussian():
#     print("=== Testing GaussianDistribution ===")

#     dist = GaussianDistribution(mean=5.0, var=4.0)  # std = 2.0

#     # Test sampling
#     samples = dist.sample(10000)
#     print("Sample mean:", np.mean(samples))
#     print("Sample var :", np.var(samples))

#     # Test PDF at known positions
#     print("PDF(5) =", dist.pdf(5))
#     print("PDF(0) =", dist.pdf(0))

#     print()


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
