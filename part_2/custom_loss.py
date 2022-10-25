import lightgbm as lgb
import numpy as np


class AsymmetricLoss:
    """Class that provides asymmetric custom loss functions for LightGBM"""

    def __init__(self, penalty=2):
        self.penalty = penalty

    def custom_asymmetric_train(self, y_pred: list, dataset_true: lgb.Dataset) -> tuple:
        """Custom loss function for model training, penalizes underforecasting

        Args:
            y_pred (list): predictions
            dataset_true (lgb.Dataset): LGBM dataset containing true values

        Returns:
            tuple: gradient and hessian
        """
        y_true = dataset_true.get_label()
        residual = (y_true - y_pred).astype("float")
        grad = np.where(residual > 0, -2 * self.penalty * residual, -2 * residual)
        hess = np.where(residual > 0, 2 * self.penalty, 2.0)

        return grad, hess

    def custom_asymmetric_valid(self, y_pred: list, dataset_true: lgb.Dataset) -> tuple:
        """Custom loss function for model validation, penalizes underforecasting

        Args:
            y_pred (list):  predictions
            dataset_true (lgb.Dataset): LGBM dataset containing true values

        Returns:
            tuple: name of metric, loss, boolean indicating whether higher is better
        """
        y_true = dataset_true.get_label()
        residual = (y_true - y_pred).astype("float")
        loss = np.where(residual > 0, (residual**2) * self.penalty, residual**2)
        
        return "custom_asymmetric_eval", np.mean(loss), False
