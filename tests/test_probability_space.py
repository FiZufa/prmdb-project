import numpy as np
from pmrdb.probability_space import ProbabilitySpace


def test_probability_space_basic():
    print("=== TEST 1: Basic Hypothesis Update ===")

    space = ProbabilitySpace()

    # hypotheses and priors
    space.set_priors({"UP": 0.6, "DOWN": 0.4})

    # Gaussian likelihoods for modality "trajectory"
    mean_up, std_up = 0.0, 1.0
    mean_down, std_down = 3.0, 1.0

    def likelihood_traj(x, hyp):
        if hyp == "UP":
            return space.gaussian_likelihood(x, mean_up, std_up)
        else:
            return space.gaussian_likelihood(x, mean_down, std_down)

    space.register_likelihood("trajectory", likelihood_traj)

    # Evidence closer to UP
    evidence = {"trajectory": 0.2}

    post = space.posterior(evidence)
    print("Posterior:", post)

    assert post["UP"] > post["DOWN"]
    print("PASSED\n")


def test_probability_space_multimodal():
    print("=== TEST 2: Multimodal Fusion ===")

    space = ProbabilitySpace()

    space.set_priors({"UP": 0.5, "DOWN": 0.5})

    # Gaussian for "trajectory"
    def like_traj(x, h):
        return space.gaussian_likelihood(x, 0 if h == "UP" else 3, 1)

    # Gaussian for "volatility"
    def like_vol(x, h):
        return space.gaussian_likelihood(x, 1 if h == "UP" else -1, 1)

    space.register_likelihood("trajectory", like_traj)
    space.register_likelihood("volatility", like_vol)

    evidence = {"trajectory": 0.1, "volatility": 1.2}

    post = space.posterior(evidence)
    print("Posterior:", post)

    assert post["UP"] > 0.8    # should strongly favor UP
    print("PASSED\n")


def test_probability_space_uniform_fallback():
    print("=== TEST 3: Zero-likelihood Fallback ===")

    space = ProbabilitySpace()
    space.set_priors({"A": 0.5, "B": 0.5})

    # Likelihood always zero → degeneracy
    def zero_fn(x, h):
        return 0.0

    space.register_likelihood("dummy", zero_fn)

    post = space.posterior({"dummy": 123})
    print("Posterior:", post)

    assert abs(post["A"] - 0.5) < 1e-9
    assert abs(post["B"] - 0.5) < 1e-9
    print("PASSED\n")


if __name__ == "__main__":
    test_probability_space_basic()
    test_probability_space_multimodal()
    test_probability_space_uniform_fallback()
