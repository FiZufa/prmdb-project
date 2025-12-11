# synthetic_data/datasets.py

import numpy as np

class SyntheticMultimodalExample:
    """
    Holds a single synthetic multimodal data record.
    
    Attributes:
        trajectory: Gaussian analogue summary (mean, variance)
        regime: categorical regime embedding (Dirichlet)
        volatility: Gaussian distribution for volatility signal
        label: true class (0 = down, 1 = up)
    """
    def __init__(self, trajectory, regime, volatility, label):
        self.trajectory = trajectory        # dict: {"mean": float, "var": float}
        self.regime = regime                # dict: {"alpha": np.array}
        self.volatility = volatility        # dict: {"mean": float, "var": float}
        self.label = label                  # int
    
    def to_dict(self):
        """Convert to JSON-serializable format."""
        return {
            "trajectory": self.trajectory,
            "regime": self.regime,
            "volatility": self.volatility,
            "label": self.label,
        }

    def __repr__(self):
        return f"SyntheticMultimodalExample(label={self.label})"
