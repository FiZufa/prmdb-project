import numpy as np


class ProbabilitySpace:
    """
    Full Bayesian probability space for PMRDB.

    Supports:
    - hypotheses (Y)
    - per-modality likelihood models
    - priors P(Y)
    - multimodal evidence P(T, R, V | Y)
    - normalized posterior P(Y | evidence)
    """

    def __init__(self, modalities=None):
        # Names of modalities (trajectory, regime, volatility, etc.)
        self.modalities = modalities if modalities else []

        # Hypotheses: dict name → prior probability
        self.priors = {}

        # Likelihood functions: modality -> function(evidence, hypothesis)
        self.likelihood_functions = {}

    # ----------------------------------------------------------
    # HYPOTHESES / PRIORS
    # ----------------------------------------------------------
    def set_priors(self, priors: dict):
        """
        priors: dict { hypothesis_name: probability }
        """
        # normalize automatically
        Z = sum(priors.values())
        self.priors = {k: v / Z for k, v in priors.items()}

    def add_hypothesis(self, name, prior):
        self.priors[name] = prior
        # normalize
        Z = sum(self.priors.values())
        for k in self.priors:
            self.priors[k] /= Z

    # ----------------------------------------------------------
    # LIKELIHOOD MODELS
    # ----------------------------------------------------------
    def register_likelihood(self, modality: str, func):
        """
        func must follow:   f(evidence_value, hypothesis) -> P(x | Y=h)
        """
        self.likelihood_functions[modality] = func
        if modality not in self.modalities:
            self.modalities.append(modality)

    # ----------------------------------------------------------
    # JOINT LIKELIHOOD
    # ----------------------------------------------------------
    def joint_likelihood(self, evidence_dict: dict, hypothesis: str):
        """
        Computes P(evidence | hypothesis) as a product of
        modality-wise likelihoods.
        """
        prob = 1.0

        for modality, evidence in evidence_dict.items():
            if modality not in self.likelihood_functions:
                raise ValueError(f"No likelihood registered for modality: {modality}")

            likelihood_fn = self.likelihood_functions[modality]
            prob *= likelihood_fn(evidence, hypothesis)

        return prob

    # ----------------------------------------------------------
    # POSTERIOR INFERENCE
    # ----------------------------------------------------------
    def posterior(self, evidence_dict: dict):
        """
        Computes posterior distribution:
            P(Y | evidence) ∝ P(evidence | Y) * P(Y)
        """

        numerators = {}
        for hypothesis, pY in self.priors.items():
            likelihood = self.joint_likelihood(evidence_dict, hypothesis)
            numerators[hypothesis] = likelihood * pY

        # normalization constant
        Z = sum(numerators.values())

        if Z == 0:
            # fallback: uniform distribution
            n = len(numerators)
            return {h: 1 / n for h in numerators}

        return {h: numerators[h] / Z for h in numerators}
    
    def register_prior(self, hypothesis, prior_value):
        """
        Register a prior probability for a hypothesis.
        """
        self.priors[hypothesis] = prior_value


    # ----------------------------------------------------------
    # Built-in distribution wrappers
    # ----------------------------------------------------------
    @staticmethod
    def gaussian_likelihood(x, mean, std):
        coeff = 1 / (np.sqrt(2 * np.pi) * std)
        exponent = np.exp(-0.5 * ((x - mean) / std) ** 2)
        return float(coeff * exponent)

    @staticmethod
    def beta_likelihood(x, a, b):
        from scipy.stats import beta
        return float(beta.pdf(x, a, b))

    @staticmethod
    def dirichlet_likelihood(vec, alpha_vec):
        from scipy.stats import dirichlet
        return float(dirichlet.pdf(vec, alpha_vec))
