
## 🧩 Module Responsibilities in the PMR-DB Workflow

| Module | Responsibility | Mathematical Role |
|------|----------------|-------------------|
| **distributions.py** | Implements parametric probability distributions (Gaussian, Beta, Dirichlet) and sampling utilities | Defines likelihood models P(X I H) and prior distributions P(H) |
| **probability_space.py** | Manages hypotheses, priors, modality-wise likelihood functions, and posterior inference | Stores P(H), computes joint likelihood P(X I H), and applies Bayes’ theorem |
| **fusion.py** | Combines evidence from multiple modalities and distributions | Performs probabilistic fusion: P(H I X) |
| **pmrdb.py** | Orchestrates the full inference pipeline end-to-end | Executes Bayesian reasoning, uncertainty aggregation, and final hypothesis selection |

## Workflow
