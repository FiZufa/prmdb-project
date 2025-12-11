# pmrdb/fusion.py

import numpy as np

# ----------------------------
# FUSION OF GAUSSIANS
# ----------------------------
def fuse_gaussians(gaussians):
    precisions = [1.0 / g.var for g in gaussians]
    fused_var = 1.0 / sum(precisions)

    weighted_means = [g.mean / g.var for g in gaussians]
    fused_mean = fused_var * sum(weighted_means)

    G = gaussians[0].__class__
    return G(fused_mean, fused_var)


# ----------------------------
# FUSION OF DIRICHLET
# ----------------------------
def fuse_dirichlet(dirichlets):
    alphas = [d.alpha for d in dirichlets]
    fused_alpha = np.sum(alphas, axis=0)

    D = dirichlets[0].__class__
    return D(fused_alpha)


# ----------------------------
# MULTIMODAL FUSION
# ----------------------------
def multimodal_fusion(modality_posteriors):
    modalities = list(modality_posteriors.keys())
    weights = np.array([modality_posteriors[m]["weight"] for m in modalities])
    weights = weights / np.sum(weights)

    pdfs = [modality_posteriors[m]["dist"].pdf for m in modalities]

    def fused_pdf(x):
        val = 1.0
        for w, pdf in zip(weights, pdfs):
            val *= pdf(x) ** w
        return val

    return fused_pdf


# ======================================================================
# EVIDENCE FUSION CLASS (needed by pmrdb.py)
# ======================================================================

class EvidenceFusion:
    """
    Simple wrapper to expose static fusion utilities
    so that pmrdb.py can import EvidenceFusion.
    """

    @staticmethod
    def fuse_gaussians(gaussians):
        return fuse_gaussians(gaussians)

    @staticmethod
    def fuse_dirichlet(dirichlets):
        return fuse_dirichlet(dirichlets)

    @staticmethod
    def fuse_multimodal(posteriors):
        return multimodal_fusion(posteriors)
