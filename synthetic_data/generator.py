# synthetic_data/generator.py

import numpy as np
from datasets import SyntheticMultimodalExample


class SyntheticDataGenerator:
    """
    Generates synthetic multimodal data for testing PMR-DB.
    
    Modalities:
        - trajectory analogues (Gaussian)
        - regime embeddings (Dirichlet)
        - volatility/momentum signals (Gaussian)
    
    Each modality correlates with the true label so that 
    Bayesian updates can be evaluated.
    """

    def __init__(self,
                 n_samples=500,
                 noise_level=0.1,
                 seed=42):
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    #    Helper: Generate trajectory analogue Gaussian signals
    # ------------------------------------------------------------------ #
    def _generate_trajectory(self, label):
        """
        If label = 1 (UP): trajectory mean is positive.
        If label = 0 (DOWN): trajectory mean is negative.
        """
        mean = (0.5 if label == 1 else -0.5) + self.rng.normal(0, 0.2)
        var = 0.1 + self.rng.normal(0, self.noise_level) ** 2
        return {"mean": mean, "var": var}

    # ------------------------------------------------------------------ #
    #    Helper: Generate regime Dirichlet distribution
    # ------------------------------------------------------------------ #
    def _generate_regime(self, label):
        """
        Encode regime likelihood via Dirichlet parameters:
            UP → alpha biased toward [trend-up]
            DOWN → alpha biased toward [trend-down]
        """
        if label == 1:
            base = np.array([3.0, 1.0, 1.0])
        else:
            base = np.array([1.0, 3.0, 1.0])

        noise = self.rng.normal(0, self.noise_level, size=3)
        alpha = np.abs(base + noise)
        return {"alpha": alpha}

    # ------------------------------------------------------------------ #
    #    Helper: Generate volatility Gaussian signals
    # ------------------------------------------------------------------ #
    def _generate_volatility(self, label):
        """
        For UP: momentum tends to positive.
        For DOWN: momentum tends to negative.
        """
        momentum_mean = (0.4 if label == 1 else -0.4) + self.rng.normal(0, 0.2)
        momentum_var = 0.05 + np.abs(self.rng.normal(0, self.noise_level)) ** 2
        
        return {"mean": momentum_mean, "var": momentum_var}

    # ------------------------------------------------------------------ #
    #    Main method: Generate dataset
    # ------------------------------------------------------------------ #
    def generate(self):
        """
        Returns list of SyntheticMultimodalExample objects.
        """
        synthetic_data = []

        for _ in range(self.n_samples):
            label = self.rng.integers(0, 2)  # 0 = down, 1 = up

            trajectory = self._generate_trajectory(label)
            regime = self._generate_regime(label)
            vol = self._generate_volatility(label)

            sample = SyntheticMultimodalExample(
                trajectory=trajectory,
                regime=regime,
                volatility=vol,
                label=label
            )

            synthetic_data.append(sample)

        return synthetic_data
