

## Module Summary
- `distributions.py`: 
    - Define probability distributions and fitting.
    - Implement(X|H) and P(H).
    - Defines parametrics PDFs.
- `probability_space.py`: Stores priors, likelihoods, and evidence.
- `fusion.py`: Compute bayesian posterior P(H|X).
- `pmrdb.py`: High-level interface for PMR-DB, executing pipelines.
    Responsibilities:
    - Register labels
    - Register priors
    - Register likelihood distributions
    - Fit all likelihood using label data
    - Evaluate posterior for a new observation

## Workflow
