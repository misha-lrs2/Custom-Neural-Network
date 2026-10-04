import pandas as pd
import numpy as np
from typing import Callable, Tuple


class LineaireRegressie:
    """
    Linear regression model trained with gradient descent.

    Supports simple linear regression (1 feature) and multiple linear
    regression (2 features).

    Attributes:
        a1 (float): Learned slope for the first feature (x1). None before training.
        a2 (float): Learned slope for the second feature (x2). None before training or for 1 feature.
        b (float): Learned intercept. None before training.
    """
    a1 = None
    a2 = None
    b = None

    def _gradient_descent_2d(
        self,
        calc_grads: Callable[[float, float], Tuple[float, float]],
        a10: float,
        b0: float,
        lr: float = 0.1,
        num_steps: int = 500,
        stop_criteria: float = 0.01,
    ):
        """
        Internal gradient descent engine for 1 feature (optimizes a1 and b).

        Args:
            calc_grads: Function computing the partial derivatives of the loss w.r.t. a1 and b.
            a10: Initial value for slope a1.
            b0: Initial value for intercept b.
            lr: Learning rate (step size).
            num_steps: Maximum number of iterations.
            stop_criteria: Threshold below which gradients are treated as flat and the loop stops.

        Returns:
            Updates the instance attributes directly.
        """
        a1_old = a10
        b_old = b0

        for _ in range(num_steps):
            gradient_a1, gradient_b = calc_grads(a1_old, b_old)

            if abs(gradient_a1) < stop_criteria and abs(gradient_b) < stop_criteria:
                break

            a1_old -= lr * gradient_a1
            b_old -= lr * gradient_b

        self.a1 = a1_old
        self.b = b_old

    def _gradient_descent_3d(
        self,
        calc_grads: Callable[[float, float, float], Tuple[float, float, float]],
        a10: float,
        a20: float,
        b0: float,
        lr: float = 0.05,
        num_steps: int = 5000,
        stop_criteria: float = 0.001,
    ):
        """
        Internal gradient descent engine for 2 features (optimizes a1, a2, and b).

        Args:
            calc_grads: Function computing the partial derivatives of the loss w.r.t. a1, a2, and b.
            a10: Initial value for slope a1.
            a20: Initial value for slope a2.
            b0: Initial value for intercept b.
            lr: Learning rate (step size).
            num_steps: Maximum number of iterations.
            stop_criteria: Threshold below which gradients are treated as flat and the loop stops.

        Returns:
            Updates the instance attributes directly.
        """
        a1_old = a10
        a2_old = a20
        b_old = b0

        for _ in range(num_steps):
            gradient_a1, gradient_a2, gradient_b = calc_grads(a1_old, a2_old, b_old)

            if (
                abs(gradient_a1) < stop_criteria
                and abs(gradient_a2) < stop_criteria
                and abs(gradient_b) < stop_criteria
            ):
                break

            a1_old -= lr * gradient_a1
            a2_old -= lr * gradient_a2
            b_old -= lr * gradient_b

        self.a1 = a1_old
        self.a2 = a2_old
        self.b = b_old

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        a10: float = 0,
        a20: float = 0,
        b0: float = 5,
        lr: float = 0.05,
        num_steps: int = 500,
        stop_criteria: float = 0.001,
    ):
        """
        Train the regression model on the given dataset.

        Automatically selects 1D or 2D optimization based on the number of
        columns in X, defines MSE partial derivatives, and runs iterative optimization.

        Args:
            X: Input features for training (maximum 2 columns).
            y: Actual target values (labels).
            a10: Initial guess for slope a1. Default 0.
            a20: Initial guess for slope a2. Default 0.
            b0: Initial guess for intercept b. Default 5.
            lr: Learning rate. Default 0.05.
            num_steps: Maximum number of iterations. Default 500.
            stop_criteria: Gradient stop threshold. Default 0.001.

        Raises:
            ValueError: If X contains more than 2 columns.
        """
        cols = len(X.columns)
        # Ensure y is a fast numpy array
        y_arr = np.asarray(y)

        if cols == 1:
            x1 = X.iloc[:, 0].to_numpy()

            def calc_grads(a1, b):
                # Calculate the error vector ONCE per step
                error = (a1 * x1 + b) - y_arr
                return np.mean(2 * error * x1), np.mean(2 * error)

            self._gradient_descent_2d(calc_grads, a10, b0, lr, num_steps, stop_criteria)

        elif cols == 2:
            x1 = X.iloc[:, 0].to_numpy()
            x2 = X.iloc[:, 1].to_numpy()

            def calc_grads(a1, a2, b):
                # Calculate the error vector ONCE per step
                error = (a1 * x1 + a2 * x2 + b) - y_arr
                return np.mean(2 * error * x1), np.mean(2 * error * x2), np.mean(2 * error)

            self._gradient_descent_3d(calc_grads, a10, a20, b0, lr, num_steps, stop_criteria)

        else:
            raise ValueError("This model supports only one or two feature columns.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values for new data using the trained model.

        Args:
            X: Feature dataset for prediction. Must have the same number of
               columns as the training data.

        Returns:
            Array of predicted y values for each row in X.

        Raises:
            ValueError: If X contains more than 2 columns.
        """
        cols = len(X.columns)

        if cols == 1:
            new_x1 = X.iloc[:, 0].to_numpy()
            predictions = self.a1 * new_x1 + self.b
            return np.array(predictions)

        elif cols == 2:
            new_x1 = X.iloc[:, 0].to_numpy()
            new_x2 = X.iloc[:, 1].to_numpy()
            predictions = self.a1 * new_x1 + self.a2 * new_x2 + self.b
            return np.array(predictions)

        else:
            raise ValueError("This model supports only one or two feature columns.")

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """
        Evaluate model accuracy via Root Mean Squared Error (RMSE).

        Args:
            X: Feature dataset.
            y: Actual target values.

        Returns:
            Computed RMSE (lower is better).
        """
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y_pred - np.asarray(y)) ** 2))
        return rmse