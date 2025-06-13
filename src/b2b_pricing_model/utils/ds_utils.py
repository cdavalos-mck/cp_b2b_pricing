import logging
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    KFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

MODELS = {"lightgbm": lgb.LGBMRegressor, "catboost": CatBoostRegressor}
METRICS = {
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
}


class ModelOptimizer:
    """
    A class for optimizing hyperparameters of regression models using Optuna.
    """

    def __init__(self, params: dict, random_state: int = 42):
        self.categorical_features = params["features"]["categorical"]
        if not self.categorical_features:
            logger.warning(
                "No categorical features provided. Using default empty list."
            )
            self.categorical_features = []
        self.numerical_features = params.get("features", {}).get("numerical", {}).get(
            "comma_columns", []
        ) + params.get("features", {}).get("numerical", {}).get("percent_columns", [])
        self.target_feature = params.get("features", {}).get("target", None)
        self.optimize_hyperparams = params.get("optimize_hyperparams", False)
        self.algorithms = params.get("algorithms", [])
        self.hyperparameters = params.get("hyperparameters", {})
        self.random_state = random_state
        self.results = {}

    def get_model_performance(
        self,
        data: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """
        Get the performance of regression algorithms with hyperparameter optimization.

        Args:
            params (dict): Configuration parameters including algorithms and hyperparameters.
            X (pd.DataFrame): Features including cluster column if using group-based CV.
            y (pd.Series): Target variable.

        Returns:
            dict: Dictionary containing the best parameters and scores for each algorithm.
        """

        data = data[data[self.target_feature] > 0]

        X = data[self.numerical_features + self.categorical_features]
        y = data[self.target_feature]

        if not self.optimize_hyperparams:
            logger.warning("Hyperparameter optimization is disabled")
            return {}

        if not self.algorithms:
            raise ValueError("No algorithms specified in parameters")

        results = {}

        for algorithm_name in self.algorithms:
            if algorithm_name not in MODELS:
                logger.warning(f"Unknown algorithm: {algorithm_name}. Skipping...")
                continue

            logger.info(f"Optimizing {algorithm_name}...")

            try:
                best_params, best_score, study = self._get_best_params(
                    algorithm_name=algorithm_name,
                    X=X,
                    y=y,
                )

                results[algorithm_name] = {
                    "best_params": best_params,
                    "best_score": best_score,
                    "n_trials": len(study.trials),
                }

                logger.info(
                    f"{algorithm_name} optimization completed. Best score: {best_score:.4f}"
                )

            except Exception as e:
                logger.error(f"Error optimizing {algorithm_name}: {str(e)}")
                continue

        return results

    def _get_best_params(
        self,
        algorithm_name: str,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[dict[str, Any], float, optuna.Study]:
        """
        Find the best hyperparameters for a specific algorithm using Optuna.

        Args:
            params (dict): Configuration parameters.
            algorithm_name (str): Name of the algorithm to optimize.
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.

        Returns:
            tuple: Best parameters, best score, and the study object.
        """

        n_trials = self.hyperparameters.get("n_trials", 100)
        n_splits = self.hyperparameters.get("n_splits", 5)
        metric = self.hyperparameters.get("metric", "rmse")

        # Validate metric
        if metric not in METRICS:
            logger.warning(f"Unknown metric: {metric}. Using RMSE instead.")
            metric = "rmse"

        algorithm_params = self.hyperparameters.get(algorithm_name, {})

        # Handle group-based cross-validation
        X_train = X.copy()

        revenue = X["c_yearly_network_volumen"]  # * y

        # weight = np.log1p(revenue)
        weight = revenue / revenue.sum()

        def objective(trial):
            """Objective function for Optuna optimization."""

            # Get base parameters for the algorithm
            params_config = _get_base_params(
                algorithm_name=algorithm_name, random_state=self.random_state
            )

            # Add hyperparameters based on configuration
            for param_name, config in algorithm_params.items():
                param_value = self._suggest_parameter(trial, param_name, config)
                params_config[param_name] = param_value

            # Create the model pipeline
            pipeline = _create_pipeline(
                algorithm_name=algorithm_name,
                algorithm_params=params_config,
                categorical_features=self.categorical_features,
                numerical_features=self.numerical_features,
            )

            preprocessor = pipeline.named_steps["preprocessor"]
            preprocessor.fit(X_train)

            feature_names = preprocessor.get_feature_names_out()

            monotone_constraints = [0] * len(feature_names)
            for idx, fname in enumerate(feature_names):
                if fname.endswith(
                    "c_yearly_network_volumen"
                ):  # or fname == "num__volume" if you know the prefix
                    monotone_constraints[idx] = 0

            # if algorithm_name.lower() == "lightgbm":
            #     params_config["monotone_constraints"] = monotone_constraints
            # elif algorithm_name.lower() == "catboost":
            #     params_config["monotone_constraints"] = monotone_constraints

            pipeline = _create_pipeline(
                algorithm_name=algorithm_name,
                algorithm_params=params_config,
                categorical_features=self.categorical_features,
                numerical_features=self.numerical_features,
            )

            # group = X_train["cluster"]
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            # sgkf = StratifiedKFold(
            #     n_splits=n_splits, shuffle=True, random_state=self.random_state
            # )
            splits = list(cv.split(X_train, y))
            # splits = list(cv.split(X_train, y, groups=group))
            # splits = list(sgkf.split(X, group))

            # Perform cross-validation
            try:
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y,
                    cv=splits,
                    scoring=METRICS[metric],
                    fit_params={"regressor__sample_weight": weight},
                    n_jobs=-1,
                    error_score="raise",
                )

                # Return the mean score (note: sklearn returns negative scores for error metrics)
                return -np.mean(cv_scores)

            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return float("inf")  # Return worst possible score for failed trials

        # Create and run the study
        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=self.random_state)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[self._optuna_callback],
        )

        return study.best_params, study.best_value, study

    def _suggest_parameter(self, trial, param_name: str, config: dict[str, Any]) -> Any:
        """Suggest a parameter value based on its configuration."""

        param_type = config.get("type")

        if param_type == "int":
            return trial.suggest_int(param_name, config["low"], config["high"])
        elif param_type == "float":
            log_scale = config.get("log", False)
            return trial.suggest_float(
                param_name, config["low"], config["high"], log=log_scale
            )
        elif param_type == "categorical":
            return trial.suggest_categorical(param_name, config["choices"])
        elif param_type == "fixed":
            return config["value"]
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _optuna_callback(self, study, trial):
        """Callback function for Optuna to log progress."""
        if trial.number % 10 == 0:
            logger.info(
                f"Trial {trial.number}: Best score so far: {study.best_value:.4f}"
            )


def _get_base_params(algorithm_name: str, random_state: int) -> dict[str, Any]:
    """Get base parameters for each algorithm."""

    if algorithm_name == "lightgbm":
        return {
            "objective": "regression",
            "metric": "mae",  # Changed from mape to rmse for consistency
            "boosting_type": "gbdt",
            "verbose": -1,
            "random_state": random_state,
            "n_jobs": -1,
        }
    elif algorithm_name == "catboost":
        return {
            "loss_function": "MAE",  # Changed from MAPE to RMSE for consistency
            "eval_metric": "MAE",
            "verbose": False,
            "random_seed": random_state,
            "thread_count": -1,
        }
    else:
        return {}


# def _get_base_params(algorithm_name: str, random_state: int) -> dict[str, Any]:
#     """Get base parameters for each algorithm."""
#     alpha = 0.05  # Default quantile level for quantile regression
#     loss_function = f"Quantile:alpha={alpha}"
#     if algorithm_name == "lightgbm":
#         return {
#             "objective": "quantile",
#             "metric": "quantile",
#             "alpha": alpha,  # Set quantile level
#             "boosting_type": "gbdt",
#             "verbose": -1,
#             "random_state": random_state,
#             "n_jobs": -1,
#         }
#     elif algorithm_name == "catboost":
#         return {
#             "loss_function": loss_function,
#             "eval_metric": loss_function,
#             "verbose": False,
#             "random_seed": random_state,
#             "thread_count": -1,
#         }
#     else:
#         return {}


def _create_pipeline(
    algorithm_name: str,
    algorithm_params: dict[str, Any],
    numerical_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
) -> Pipeline:
    """
    Create a scikit-learn pipeline for the specified algorithm with given parameters.

    Args:
        algorithm_name (str): Name of the algorithm.
        params (dict): Parameters for the algorithm.

    Returns:
        Pipeline: A scikit-learn pipeline with the specified regressor.
    """
    preprocessor = _create_preprocessing_pipeline(
        numerical_features=numerical_features, categorical_features=categorical_features
    )
    # LightGBM Pipeline
    model_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", MODELS[algorithm_name](**algorithm_params)),
        ]
    )

    return model_pipeline


def _create_preprocessing_pipeline(
    numerical_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
):
    transformers = []

    # Add numerical features transformer if provided
    if numerical_features:
        transformers.append(("scaler", StandardScaler(), numerical_features))

    # Add categorical features transformer if provided
    if categorical_features:
        transformers.append(
            (
                "encoder",
                OneHotEncoder(
                    # drop=None,
                    handle_unknown="ignore",
                    # min_frequency=0.01,
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",  # Keep any remaining columns unchanged
    )

    return preprocessor
