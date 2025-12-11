import numpy as np
from pmrdb.pmrdb import PMRDB
from pmrdb.distributions import GaussianDistribution
from scipy.stats import norm

# ----------------------------------------------------------
# 1. Generate synthetic data
# ----------------------------------------------------------

def generate_synthetic_trajectories(n=200, length=40):
    print("Generating synthetic dataset...")

    X = []
    y = []

    for _ in range(n):
        label = "UP" if np.random.rand() > 0.5 else "DOWN"

        if label == "UP":
            base = np.linspace(0, 1, length)
            noise = np.random.normal(0, 0.1, length)
            traj = base + noise
        else:
            base = np.linspace(1, 0, length)
            noise = np.random.normal(0, 0.1, length)
            traj = base + noise

        X.append(traj)
        y.append(label)

    return np.array(X), np.array(y)

# ----------------------------------------------------------
# 2. Build PMRDB model
# ----------------------------------------------------------

def build_pmrdb_model():
    print("Building PMRDB model...")

    db = PMRDB()

    # Define hypotheses
    db.space.set_priors({"UP": 0.5, "DOWN": 0.5})

    return db

# ----------------------------------------------------------
# 3. Fit likelihood distributions
# ----------------------------------------------------------

def fit_distributions(db, X, y):
    print("Fitting distributions...")

    # Extract trajectory feature (mean slope)
    slopes = np.array([np.mean(np.diff(x)) for x in X])

    slopes_UP   = slopes[y == "UP"]
    slopes_DOWN = slopes[y == "DOWN"]

    # Fit 1D Gaussian for slope
    dist_UP = GaussianDistribution(mean=np.mean(slopes_UP),
                                   var=np.var(slopes_UP) + 1e-6)

    dist_DOWN = GaussianDistribution(mean=np.mean(slopes_DOWN),
                                     var=np.var(slopes_DOWN) + 1e-6)

    # Register likelihood function
    db.space.register_likelihood(
        "trajectory",
        lambda x, h: dist_UP.pdf(x) if h == "UP" else dist_DOWN.pdf(x)
    )

# ----------------------------------------------------------
# 4. Evaluate PMRDB
# ----------------------------------------------------------

def evaluate(db, X, y):
    print("Running evaluation...")

    slopes = np.array([np.mean(np.diff(x)) for x in X])

    correct = 0
    total = len(X)

    for i in range(total):
        obs = {"trajectory": slopes[i]}

        posterior = db.space.posterior(obs)

        pred = max(posterior, key=posterior.get)

        if pred == y[i]:
            correct += 1

    acc = correct / total
    print(f"Accuracy: {acc:.3f}")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

if __name__ == "__main__":
    X, y = generate_synthetic_trajectories()

    db = build_pmrdb_model()

    fit_distributions(db, X, y)

    evaluate(db, X, y)
