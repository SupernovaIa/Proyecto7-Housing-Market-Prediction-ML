# Data processing  
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualization  
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Model selection and evaluation  
# -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold

# Regression models  
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics  
# -----------------------------------------------------------------------
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Preprocessing  
# -----------------------------------------------------------------------
from sklearn.preprocessing import KBinsDiscretizer

# XGBoost  
# -----------------------------------------------------------------------
import xgboost as xgb


def df_results(real, predicted, dataset, model):
    """
    Creates a DataFrame summarizing the results of a model's predictions.

    Parameters:
    - real (array-like): The actual values from the dataset.
    - predicted (array-like): The predicted values from the model.
    - dataset (str): The name or identifier of the dataset used.
    - model (str): The name or identifier of the model used.

    Returns:
    - (pd.DataFrame): A DataFrame containing columns for real values, predicted values, dataset, model, and residuals (calculated as `real - predicted`).
    """
    return pd.DataFrame({
        "real": real,
        "predicted": predicted,
        "dataset": dataset,
        "model": model,
        "residuals": real - predicted,
    })


class RegressionModels:
    """
    A class for managing regression models, including dataset preparation, model training, evaluation, and visualization.

    Attributes:
    - df (pd.DataFrame): The input DataFrame containing features and the target variable.
    - target_variable (str): The name of the target variable column in the DataFrame.
    - X (pd.DataFrame): The feature matrix after dropping the target variable.
    - y (pd.Series): The target variable values.
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Test set features.
    - y_train (pd.Series): Training set target values.
    - y_test (pd.Series): Test set target values.
    - models (dict): A dictionary of initialized models including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
    - predictions (dict): A dictionary to store predictions for each model on training and test sets.
    - best_model (dict): A dictionary to store the best model for each algorithm.
    - results (pd.DataFrame or None): A DataFrame containing model predictions and residuals or None if no results are available.
    """

    def __init__(self, df, target_variable, frac=0.8, seed=42):
        """
        Initializes an object for managing dataset splitting, model training, and evaluation.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing features and the target variable.
        - target_variable (str): The name of the target variable column in the DataFrame.
        - frac (float, optional): The fraction of the dataset to use for training. Defaults to 0.8.
        - seed (int, optional): The random seed for reproducibility during dataset splitting. Defaults to 42.
        """

        self.df = df
        self.target_variable = target_variable
        self.X = df.drop(target_variable, axis=1)
        self.y = df[target_variable]

        # Dataset division
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=frac, random_state=seed, shuffle=True)

        # Model configuration
        self.models = {
            "linear": LinearRegression(n_jobs=-1),
            "tree": DecisionTreeRegressor(),
            "random_forest": RandomForestRegressor(),
            "gradient_boosting": GradientBoostingRegressor(),
            "xgboost": xgb.XGBRegressor(),
        }

        # Prediction and best models storage
        self.predictions = {model: {"train": None, "test": None} for model in self.models}
        self.best_model = {model: None for model in self.models}
        self.results = None


    def model_fit(self, model, param_grid=None):
        """
        Fits a specified model to the training data, optionally performing hyperparameter tuning.

        Parameters:
        - model (str): The name of the model to fit. Must be a key in the `self.models` dictionary.
        - param_grid (dict, optional): A dictionary specifying hyperparameter values for GridSearchCV. If provided, hyperparameter tuning is performed. Defaults to None.

        Raises:
        - ValueError: If the specified model name is not recognized.

        Returns:
        - (dict, optional): If `param_grid` is provided, returns the cross-validation results from GridSearchCV.
        """

        if model not in self.models:
            raise ValueError(f"Modelo '{model}' no reconocido.")

        estimator = self.models[model]

        if param_grid:
            grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model[model] = grid_search.best_estimator_
        else:
            estimator.fit(self.X_train, self.y_train)
            self.best_model[model] = estimator

        # Predictions
        self.predictions[model]["train"] = self.best_model[model].predict(self.X_train)
        self.predictions[model]["test"] = self.best_model[model].predict(self.X_test)

        if param_grid:
            return grid_search.cv_results_


    def get_results(self):
        """
        Generates and returns a DataFrame summarizing the predictions and residuals for all fitted models.

        Raises:
        - ValueError: If no models have been fitted and no results are available.

        Returns:
        - (pd.DataFrame): A concatenated DataFrame containing prediction results and residuals for both training and test datasets across all fitted models.
        """

        results = []

        for model, pred in self.predictions.items():
            if pred["train"] is not None and pred["test"] is not None:
                results.extend([
                    df_results(self.y_train, pred["train"], "Train", model),
                    df_results(self.y_test, pred["test"], "Test", model),
                ])

        if not results:
            raise ValueError("Must fit at least one model to get results.")
        
        self.results = pd.concat(results, axis=0)
        return self.results
    

    def get_metrics(self, model):
        """
        Calculates and returns performance metrics for a specified model on both training and test datasets.

        Parameters:
        - model (str): The name of the model for which metrics are to be calculated. Must be a key in the `self.predictions` dictionary.

        Raises:
        - ValueError: If the specified model name is not recognized.
        - ValueError: If the specified model has not been fitted.

        Returns:
        - (pd.DataFrame): A DataFrame containing R², MAE, and RMSE metrics for both training and test datasets.
        """

        if model not in self.predictions:
            raise ValueError(f"Modelo '{model}' no reconocido.")

        pred = self.predictions[model]
        if pred["train"] is None or pred["test"] is None:
            raise ValueError(f"Debe ajustar el modelo '{model}' antes de calcular métricas.")

        metrics = {
            "train": {
                "R2": r2_score(self.y_train, pred["train"]),
                "MAE": mean_absolute_error(self.y_train, pred["train"]),
                "RMSE": np.sqrt(mean_squared_error(self.y_train, pred["train"])),
            },
            "test": {
                "R2": r2_score(self.y_test, pred["test"]),
                "MAE": mean_absolute_error(self.y_test, pred["test"]),
                "RMSE": np.sqrt(mean_squared_error(self.y_test, pred["test"])),
            },
        }
        return pd.DataFrame(metrics).T
    

    def plot_residuals(self, model):
        """
        Plots the residuals for the specified model on both training and test datasets.

        Parameters:
        - model (str): The name of the model for which residual plots are to be generated.

        Raises:
        - ValueError: If results are not available and the residuals have not been computed.

        Returns:
        - None: Displays the residual plots for training and test datasets.
        """

        if self.results is None:
            raise ValueError("Must get results before plotting")

        data = self.results[self.results["model"] == model]
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(data[data["dataset"] == "Train"], x="residuals", kde=True, ax=ax[0], color="blue")
        sns.histplot(data[data["dataset"] == "Test"], x="residuals", kde=True, ax=ax[1], color="green")
        ax[0].set_title(f"Residuals - Train ({model})")
        ax[1].set_title(f"Residuals - Test ({model})")
        plt.tight_layout()
        plt.show()


    def predictors_importance(self, model):
        """
        Calculates and visualizes feature importance for tree-based models.

        Parameters:
        - model (str): The name of the tree-based model for which feature importance is to be calculated. Must be one of ["tree", "random_forest", "gradient_boosting", "xgboost"].

        Raises:
        - ValueError: If the specified model is not tree-based.
        - ValueError: If the specified model has not been fitted.

        Returns:
        - (pd.DataFrame): A DataFrame containing the predictors and their corresponding importance scores, sorted in descending order.
        """

        if model not in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
            raise ValueError("Only valid for tree-based models.")

        if self.best_model[model] is None:
            raise ValueError(f"Must fit '{model}' before getting predictors importance.")

        importances = self.best_model[model].feature_importances_
        importance_df = pd.DataFrame({"Predictor": self.X_train.columns, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Predictor", data=importance_df, palette="mako")
        plt.title(f"Predictors importance - {model}")
        plt.show()
        return importance_df
    

    def plot_actual_vs_prediction(self, model):
        """
        Plots actual vs. predicted values for the specified model on both training and test datasets.

        Parameters:
        - model (str): The name of the model for which the plot is to be generated.

        Raises:
        - ValueError: If the specified model has not been fitted.

        Returns:
        - None: Displays scatter plots of actual vs. predicted values for training and test datasets, including R² and RMSE metrics.
        """

        y_train_pred = self.best_model[model].predict(self.X_train)
        y_train = self.y_train

        y_test_pred = self.best_model[model].predict(self.X_test)
        y_test = self.y_test

        metrics = self.get_metrics(model)

        r2_train = metrics['R2']['train']
        rmse_train = metrics['RMSE']['train']
        r2_test = metrics['R2']['test']
        rmse_test = metrics['RMSE']['test']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Train
        sns.scatterplot(ax=axes[0], x=y_train, y=y_train_pred, color="dodgerblue", s=50, alpha=0.7)
        axes[0].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', lw=2)
        axes[0].set_title("Train Data", color="dodgerblue")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Predicted", fontsize=12)
        axes[0].text(0.05, 0.95, f"$R^2$: {r2_train:.2f}\nRMSE: {rmse_train:.2f}",
                    transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))

        # Test
        sns.scatterplot(ax=axes[1], x=y_test, y=y_test_pred, color="darkgreen", s=50, alpha=0.7)
        axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
        axes[1].set_title("Test Data", color="darkgreen")
        axes[1].set_xlabel("")
        axes[1].text(0.05, 0.95, f"$R^2$: {r2_test:.2f}\nRMSE: {rmse_test:.2f}",
                    transform=axes[1].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))

        fig.supxlabel("Actual Values")
        fig.suptitle(f"{model.title()} Results")

        plt.tight_layout()
        plt.show()