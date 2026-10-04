# Custom Gradient Descent Linear Regression

A from-scratch **linear regression** implementation trained via **gradient descent**, supporting univariate and bivariate models without scikit-learn's internal solvers. Validated against scikit-learn's `LinearRegression` on the classic **Auto MPG** dataset.

## Executive Summary

This project implements the full optimization loop for linear regression—computing partial derivatives of Mean Squared Error (MSE), iteratively updating slope and intercept parameters, and evaluating convergence via gradient magnitude thresholds. On the Auto MPG benchmark, the custom implementation achieves an RMSE of **4.14 MPG**, marginally outperforming scikit-learn's **4.28 MPG** on the same test split.

## Technical Overview

### Model

**Simple (1 feature):** ŷ = a₁·x₁ + b

**Multiple (2 features):** ŷ = a₁·x₁ + a₂·x₂ + b

### Gradient Descent

Partial derivatives of MSE with respect to each parameter:

```
∂C/∂a₁ = mean(2 · (ŷ − y) · x₁)
∂C/∂a₂ = mean(2 · (ŷ − y) · x₂)
∂C/∂b  = mean(2 · (ŷ − y))
```

Update rule: `θ ← θ − η · ∂C/∂θ`

Training stops when all gradient magnitudes fall below `stop_criteria` or `num_steps` is reached.

### Design Choices

| Choice | Rationale |
|--------|-----------|
| Small learning rate (0.0001) for Auto MPG | Large feature magnitudes (weight ~3500 lbs) produce steep gradients |
| Callable partial derivatives | Decouples optimization engine from specific loss formulations |
| RMSE as evaluation metric | Same scale as target (miles per gallon), directly interpretable |

### API

```python
from gradient_descent import LineaireRegressie
import pandas as pd

model = LineaireRegressie()
model.fit(X_train, y_train, lr=0.0001, num_steps=5000, stop_criteria=0.001)
predictions = model.predict(X_test)
rmse = model.score(X_test, y_test)  # Lower is better
```

## Installation & Setup

### Requirements

- Python 3.10+
- NumPy
- Pandas
- scikit-learn (benchmark comparison only)
- Matplotlib (visualization in notebook)
- Jupyter Notebook

### Setup

```bash
cd custom-gradient-descent/ai-s3-gradient-descent-misha-lrs2-main
pip install numpy pandas scikit-learn matplotlib jupyter
```

## Usage

### Run the Validation Notebook

```bash
jupyter notebook testing.ipynb
```

The notebook:
1. Loads the Auto MPG dataset (weight + displacement → MPG)
2. Splits 80/20 train/test
3. Trains both `LineaireRegressie` and scikit-learn's `LinearRegression`
4. Compares RMSE and plots predicted vs. actual values

### Programmatic Example

```python
import pandas as pd
from sklearn.datasets import fetch_openml

auto = fetch_openml(name="auto-mpg", version=1, as_frame=True, parser="auto")
df = auto.frame.dropna(subset=["mpg", "weight", "displacement"])

X = df[["weight", "displacement"]]
y = df["mpg"].astype(float)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from gradient_descent import LineaireRegressie
model = LineaireRegressie()
model.fit(X_train, y_train.values, lr=0.0001, num_steps=10000)
print(f"RMSE: {model.score(X_test, y_test.values):.2f} MPG")
print(f"Learned: a1={model.a1:.6f}, a2={model.a2:.6f}, b={model.b:.4f}")
```

## Results & Interpretation

### Auto MPG — Test Set (20% hold-out)

| Model | RMSE (MPG) |
|-------|------------|
| **Custom Gradient Descent** | ~4.14 |
| scikit-learn LinearRegression | ~4.28 |

An RMSE of 4.14 means predictions deviate from actual fuel efficiency by roughly 4 miles per gallon on average.

### Residual Analysis

Both models systematically **underestimate** MPG for the most fuel-efficient vehicles (actual MPG > 30). This reflects a fundamental limitation of linear regression: a single hyperplane cannot capture the non-linear relationship in the high-efficiency tail of the distribution.

## Project Structure

```
ai-s3-gradient-descent-misha-lrs2-main/
├── gradient_descent.py   # LineaireRegressie class
├── testing.ipynb         # Auto MPG validation notebook
└── README.md
```

## Author

**Misha Leenders** — AI & Machine Learning portfolio project (2025)
