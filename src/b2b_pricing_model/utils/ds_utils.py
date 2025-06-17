import logging
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

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

        logger.info(f"{data.columns}")
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


class RegressionModel:
    """
    A class for optimizing hyperparameters of regression models using Optuna.
    """

    def __init__(
        self, params: dict, best_hyperparameters: dict, random_state: int = 42
    ):
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
        self.best_hyperparameters = best_hyperparameters
        self.algorithms = params.get("algorithms", [])
        self.random_state = random_state
        cross_validation = params.get("cross_validation", {})
        self.n_splits = cross_validation.get("cv_folds", 5)
        self.scoring = cross_validation.get("scoring", ["rmse"])

    def train_model(self, data: pd.DataFrame):
        data = data[data[self.target_feature] > 0]
        X = data[self.numerical_features + self.categorical_features]
        y = data[self.target_feature]

        revenue = X["c_yearly_network_volumen"]  # * y

        # weight = np.log1p(revenue)
        weight = revenue / revenue.sum()

        cv_results = {}
        pipeline_list = {}

        for algorithm_name in self.algorithms:
            if algorithm_name not in MODELS:
                logger.warning(f"Unknown algorithm: {algorithm_name}. Skipping...")
                continue

            logger.info(f"Training {algorithm_name}...")

            try:
                params_config = self.best_hyperparameters[algorithm_name]["best_params"]
                pipeline = _create_pipeline(
                    algorithm_name=algorithm_name,
                    algorithm_params=params_config,
                    categorical_features=self.categorical_features,
                    numerical_features=self.numerical_features,
                )

                preprocessor = pipeline.named_steps["preprocessor"]
                preprocessor.fit(X)

                # Get transformed feature names
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

                model_cv_results = self._run_cross_validation(
                    model_pipeline=pipeline,
                    X=X,
                    y=y,
                    scoring_funcs=self.scoring,
                    weight=weight,
                )

                pipeline.fit(X, y, regressor__sample_weight=weight)

                cv_results[algorithm_name] = model_cv_results
                pipeline_list[algorithm_name] = pipeline

                logger.info(f"{algorithm_name} training completed successfully.")
            except Exception as e:
                logger.error(f"Error training {algorithm_name}: {str(e)}")
                continue

        return cv_results, pipeline_list

    def _run_cross_validation(
        self,
        model_pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        weight: np.ndarray,
        scoring_funcs: list[str] = ["rmse"],
    ) -> pd.DataFrame:
        """
        Run cross-validation on the model pipeline.

        Args:
            model_pipeline (Pipeline): The scikit-learn pipeline with the model.
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.
            n_splits (int): Number of splits for cross-validation.
            scoring_funcs (list[str]): List of scoring functions to use.

        Returns:
            pd.DataFrame: DataFrame containing cross-validation results.
        """
        # groups = X["cluster"]
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # sgkf = StratifiedKFold(
        #     n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        # )
        # splits = list(sgkf.split(X, groups))
        splits = list(cv.split(X, y))
        cv_results = cross_validate(
            model_pipeline,
            X,
            y,
            cv=splits,  # pass list of (train_idx, test_idx)
            scoring=scoring_funcs,
            fit_params={"regressor__sample_weight": weight},
            return_train_score=True,
            n_jobs=-1,
        )
        return pd.DataFrame(cv_results)


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
            "allow_writing_files": False,
            "task_type": "CPU",
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


class ClusteringPipeline:
    """
    A comprehensive pipeline for clustering analysis that includes:
    - Data preprocessing
    - Optimal cluster number selection using multiple metrics
    - Multiple clustering algorithms
    - Cluster evaluation
    - Visualization with PCA and t-SNE
    - Minimum cluster size constraints
    - k-Nearest Neighbors search for each point
    """

    def __init__(
        self,
        algorithm: str,
        min_cluster_size: int,
        k_range: range,
        random_state=42,
    ):
        self.algorithm = algorithm
        self.min_cluster_size = min_cluster_size
        self.k_range = k_range
        self.random_state = random_state
        self.preprocessor = None
        self.best_k = None
        self.best_model = None
        self.labels = None
        self.silhouette_scores = {}
        self.calinski_scores = {}
        self.davies_bouldin_scores = {}
        self.min_cluster_size_scores = {}
        self.pca = None
        self.tsne = None
        self.pca_result = None
        self.tsne_result = None
        self.X_scaled = None  # store for later use

    def preprocess_data(self, X, method="standard", **kwargs):
        if method == "standard":
            self.preprocessor = StandardScaler(**kwargs)
            self.X_scaled = self.preprocessor.fit_transform(X)
        elif method == "robust":
            self.preprocessor = RobustScaler(**kwargs)
            self.X_scaled = self.preprocessor.fit_transform(X)
        elif method == "minmax":
            # Apply feature-wise MinMaxScaler manually
            self.preprocessor = {}
            self.X_scaled = pd.DataFrame(index=X.index, columns=X.columns)
            for col in X.columns:
                scaler = MinMaxScaler(**kwargs)
                self.X_scaled[col] = scaler.fit_transform(X[[col]])
                self.preprocessor[col] = scaler
            self.X_scaled = self.X_scaled.astype(float).values
        else:
            raise ValueError("Method must be 'standard', 'robust', or 'minmax'")

        return self.X_scaled

    def get_k_nearest_neighbors(self, k=3, customer_ids=None, metric="euclidean"):
        """
        Find k-nearest neighbors for each sample in the preprocessed data.

        Parameters
        ----------
        k : int
            Number of neighbors to return (excluding the point itself).
        customer_ids : list or array-like, optional
            List of IDs corresponding to the rows in the data.
        metric : str
            Distance metric to use.

        Returns
        -------
        DataFrame with each point's neighbors and distances.
        """
        if self.X_scaled is None:
            raise ValueError("You must call preprocess_data before getting neighbors.")

        knn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        knn.fit(self.X_scaled)
        distances, indices = knn.kneighbors(self.X_scaled)

        if customer_ids is None:
            customer_ids = list(range(len(self.X_scaled)))
        customer_ids = np.array(customer_ids)

        breakpoint
        records = []
        for i, (neighbor_idxs, dists) in enumerate(zip(indices, distances)):
            src_id = customer_ids[i]
            for rank, (neighbor_idx, dist) in enumerate(
                zip(neighbor_idxs[1:], dists[1:]), start=1
            ):
                neighbor_id = customer_ids[neighbor_idx]
                records.append(
                    {
                        "customer_id": src_id,
                        "neighbor_id": neighbor_id,
                        "rank": rank,
                        "distance": dist,
                    }
                )

        return pd.DataFrame(records)

    def get_k_nearest_top_performers(
        self,
        k=3,
        customer_ids=None,
        metric="euclidean",
        performance_labels=None,
        top_label="Top Performer",
    ):
        """
        Find k-nearest neighbors for each customer, but only include neighbors that are labeled as 'Top Performer'.

        Parameters
        ----------
        k : int
            Number of top performer neighbors to return (excluding the point itself).
        customer_ids : list or array-like, optional
            List of customer IDs corresponding to self.X_scaled.
        metric : str
            Distance metric to use.
        performance_labels : list or array-like
            List of labels (same length as X_scaled), indicating performance group of each customer.
        top_label : str
            Label used to identify top performers.

        Returns
        -------
        DataFrame with each point's nearest top performer neighbors and distances.
        """
        if self.X_scaled is None:
            raise ValueError("You must call preprocess_data before getting neighbors.")
        if performance_labels is None:
            raise ValueError("You must provide performance labels for filtering.")

        knn = NearestNeighbors(
            n_neighbors=len(self.X_scaled), metric=metric
        )  # Use full set
        knn.fit(self.X_scaled)
        distances, indices = knn.kneighbors(self.X_scaled)

        if customer_ids is None:
            customer_ids = list(range(len(self.X_scaled)))
        customer_ids = np.array(customer_ids)
        performance_labels = np.array(performance_labels)

        records = []
        for i, (neighbor_idxs, dists) in enumerate(zip(indices, distances)):
            src_id = customer_ids[i]
            # Skip self index and filter only top performers
            filtered = [
                (neighbor_idx, dist)
                for neighbor_idx, dist in zip(neighbor_idxs[1:], dists[1:])
                if performance_labels[neighbor_idx] == top_label
            ][:k]  # Only top k top performers

            for rank, (neighbor_idx, dist) in enumerate(filtered, start=1):
                neighbor_id = customer_ids[neighbor_idx]
                records.append(
                    {
                        "customer_id": src_id,
                        "neighbor_id": neighbor_id,
                        "rank": rank,
                        "distance": dist,
                    }
                )

        return pd.DataFrame(records)


def explain_neighbors_original_scale(
    customer_idx, neighbor_indices, X_scaled, X_original, feature_names
):
    # Use scaled data for distance calculations
    X_scaled_pd = X_scaled.set_index("customer_id")
    X_original = X_original.set_index("customer_id")

    feature_names_extra = feature_names + [
        "c_yearly_margin_per_liter",
        "predicted_value",
        "performance_label",
    ]

    customer_scaled = X_scaled_pd.loc[customer_idx, feature_names]
    neighbors_scaled = X_scaled_pd.loc[neighbor_indices, feature_names]

    # Get original values for display
    customer_original = X_original.loc[customer_idx, feature_names]
    neighbors_original = X_original.loc[neighbor_indices, feature_names]

    # Calculate distances using scaled data
    feature_distances_scaled = np.abs(customer_scaled - neighbors_scaled)
    avg_feature_distances = np.mean(feature_distances_scaled, axis=0)

    # Create explanation with original values
    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "customer_value": customer_original,
            "avg_neighbor_value": np.mean(neighbors_original, axis=0),
            "scaled_customer_value": customer_scaled,
            "scaled_avg_neighbor_value": np.mean(neighbors_scaled, axis=0),
            "scaled_distance": avg_feature_distances,
            "value_difference": np.abs(
                customer_original - np.mean(neighbors_original, axis=0)
            ),
        }
    ).sort_values("scaled_distance")

    neighbor_indices = np.append(
        neighbor_indices, customer_idx
    )  # Include the customer itself in neighbors

    neighbors_original = X_original.loc[neighbor_indices, feature_names_extra]

    neighbors_original.reset_index(inplace=True)

    return explanation, neighbors_original
