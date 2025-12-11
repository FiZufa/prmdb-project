# tests/test_fusion.py

import numpy as np
from pmrdb.distributions import GaussianDistribution, DirichletDistribution
from pmrdb.fusion import (
    fuse_gaussians,
    fuse_dirichlet,
    multimodal_fusion,
    EvidenceFusion
)


# ---------------------------------------------------------
# TEST 1 — Gaussian Fusion
# ---------------------------------------------------------
def test_fuse_gaussians():
    print("\n=== Testing Gaussian Fusion ===")

    g1 = GaussianDistribution(mean=10, var=4)   # std = 2
    g2 = GaussianDistribution(mean=14, var=9)   # std = 3

    fused = fuse_gaussians([g1, g2])

    # Expected math:
    # precision = [1/4, 1/9]
    # fused_var = 1 / (1/4 + 1/9)
    expected_var = 1 / (1/4 + 1/9)

    # fused_mean = fused_var * (10/4 + 14/9)
    expected_mean = expected_var * (10/4 + 14/9)

    print("Fused mean:", fused.mean, "Expected:", expected_mean)
    print("Fused var :", fused.var,  "Expected:", expected_var)

    assert np.isclose(fused.mean, expected_mean)
    assert np.isclose(fused.var, expected_var)


# ---------------------------------------------------------
# TEST 2 — Dirichlet Fusion
# ---------------------------------------------------------
def test_fuse_dirichlet():
    print("\n=== Testing Dirichlet Fusion ===")

    d1 = DirichletDistribution([1, 2, 3])
    d2 = DirichletDistribution([4, 1, 0])

    fused = fuse_dirichlet([d1, d2])

    expected_alpha = np.array([5, 3, 3])  # element-wise sum

    print("Fused α:", fused.alpha, "Expected:", expected_alpha)

    assert np.allclose(fused.alpha, expected_alpha)


# ---------------------------------------------------------
# TEST 3 — Multimodal Fusion (PDF pooling)
# ---------------------------------------------------------
def test_multimodal_fusion():
    print("\n=== Testing Multimodal Fusion ===")

    g1 = GaussianDistribution(0, 1)
    g2 = GaussianDistribution(5, 4)

    modality_posteriors = {
        "A": {"dist": g1, "weight": 1.0},
        "B": {"dist": g2, "weight": 1.0},
    }

    fused_pdf = multimodal_fusion(modality_posteriors)

    # Try evaluating at a point
    x = 1.0

    val = fused_pdf(x)

    # Expected: pdf_A(x)^0.5 * pdf_B(x)^0.5 (equal weights)
    w = 0.5
    expected = g1.pdf(x)**w * g2.pdf(x)**w

    print("Fused pdf:", val, "Expected:", expected)

    assert np.isclose(val, expected)


# ---------------------------------------------------------
# TEST 4 — EvidenceFusion class wrapper
# ---------------------------------------------------------
def test_evidence_fusion_wrapper():
    print("\n=== Testing EvidenceFusion Wrapper ===")

    g1 = GaussianDistribution(0, 1)
    g2 = GaussianDistribution(5, 4)

    fused = EvidenceFusion.fuse_gaussians([g1, g2])

    # Just check type and sanity
    assert isinstance(fused, GaussianDistribution)
    assert fused.var > 0


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    test_fuse_gaussians()
    test_fuse_dirichlet()
    test_multimodal_fusion()
    test_evidence_fusion_wrapper()

    print("\nAll fusion tests completed successfully.")
