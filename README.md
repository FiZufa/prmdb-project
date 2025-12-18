
## 🧩 Module Responsibilities in the PMR-DB Workflow

| Module | Responsibility | Mathematical Role |
|------|----------------|-------------------|
| **distributions.py** | Implements parametric probability distributions (Gaussian, Beta, Dirichlet) and sampling utilities | Defines likelihood models P(X I H) and prior distributions P(H) |
| **probability_space.py** | Manages hypotheses, priors, modality-wise likelihood functions, and posterior inference | Stores P(H), computes joint likelihood P(X I H), and applies Bayes’ theorem |
| **fusion.py** | Combines evidence from multiple modalities and distributions | Performs probabilistic fusion or bayesian updates: P(H I X) |
| **pmrdb.py** | Orchestrates the full inference pipeline end-to-end | Executes Bayesian reasoning, uncertainty aggregation, and final hypothesis selection |

## 📊 Testing Module Functionalities

1. `test_distributions.py`

    Test probability distributions (Gaussian, Beta, Dirichlet) in `module distribution.py`
    ```
    === Testing GaussianDistribution ===
    Sample mean: 4.992310387960001
    Sample var : 3.969604187382648
    PDF(5) = 0.19947114020071635
    PDF(0) = 0.00876415024678427

    === Testing BetaDistribution ===
    Sample mean: 0.28419139780947145
    Sample var : 0.025171948821774266
    PDF(0.5) = 0.9374999999999999
    PDF(0.1) = 1.9682999999999997

    === Testing DirichletDistribution ===
    Sample mean vector: [0.22231246 0.33271046 0.44497708]
    Row sums (should be 1): 1.0
    PDF([0.2, 0.3, 0.5]) = 7.560000000000013
    ```

2. `test_probability_space.py`

    ```
    === TEST 1: Basic Hypothesis Update ===
    Posterior: {'UP': 0.9866850720971416, 'DOWN': 0.013314927902858318}
    PASSED

    === TEST 2: Multimodal Fusion ===
    Posterior: {'UP': 0.9986414800495711, 'DOWN': 0.001358519950428958}
    PASSED

    === TEST 3: Zero-likelihood Fallback ===
    Posterior: {'A': 0.5, 'B': 0.5}
    PASSED
    ```

3. `test_fusion.py`

    ```
    === Testing Gaussian Fusion ===
    Fused mean: 11.23076923076923 Expected: 11.23076923076923
    Fused var : 2.769230769230769 Expected: 2.769230769230769

    === Testing Dirichlet Fusion ===
    Fused α: [5 3 3] Expected: [5 3 3]

    === Testing Multimodal Fusion ===
    Fused pdf: 0.08082151101249259 Expected: 0.08082151101249259
    ```

4. `test_pmrdb.py`

    ```
    {'posterior_probability_UP': 0.5, 'uncertainty': {'mean': 0.5, 'variance': 0.0, 'aleatoric': 0.0, 'epistemic': 0.0, 'total': 0.0}}
    ```

