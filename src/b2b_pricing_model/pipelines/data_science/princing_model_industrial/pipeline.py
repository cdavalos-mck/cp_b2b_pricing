from kedro.pipeline import Pipeline, node, pipeline

from ..pricing_model.nodes import (
    get_best_hyperparameters,
    predict_first_stage,
    train_model,
)
from .nodes import create_master_base, predict_second_stage


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_master_base,
                inputs=["params:outliers_industrial", "master_industrial"],
                outputs=[
                    "regular_base_clients_industrial",
                    "regular_outlier_clients_industrial",
                ],
                name="create_master_base_industrial",
            ),
            node(
                func=get_best_hyperparameters,
                inputs=[
                    "params:first_stage_industrial",
                    "regular_base_clients_industrial",
                    "default_first_stage_best_hyperparameters_industrial",
                ],
                outputs="first_stage_best_hyperparameters_industrial",
                name="get_first_stage_best_hyperparameters_industrial",
            ),
            node(
                func=train_model,
                inputs=[
                    "params:first_stage_industrial",
                    "default_first_stage_best_hyperparameters_industrial",
                    "regular_base_clients_industrial",
                ],
                outputs=[
                    "first_stage_industrial_cv_results",
                    "first_stage_industrial_summary_df",
                    "best_first_stage_industrial_model",
                ],
                name="train_first_stage_industrial_model",
            ),
            node(
                func=predict_first_stage,
                inputs=[
                    "params:first_stage_industrial",
                    "best_first_stage_industrial_model",
                    "regular_base_clients_industrial",
                ],
                outputs=[
                    "first_stage_industrial_predictions",
                    "under_performers_industrial",
                    "top_performers_industrial",
                ],
                name="predict_first_stage_industrial",
            ),
            node(
                func=get_best_hyperparameters,
                inputs=[
                    "params:second_stage_industrial",
                    "top_performers_industrial",
                    "default_second_stage_best_hyperparameters_industrial",
                ],
                outputs="second_stage_best_hyperparameters_industrial",
                name="get_second_stage_best_hyperparameters_industrial",
            ),
            node(
                func=train_model,
                inputs=[
                    "params:second_stage_industrial",
                    "second_stage_best_hyperparameters_industrial",
                    "top_performers_industrial",
                ],
                outputs=[
                    "second_stage_industrial_cv_results",
                    "second_stage_industrial_summary_df",
                    "best_second_stage_industrial_model",
                ],
                name="train_second_stage_industrial_model",
            ),
            node(
                func=predict_second_stage,
                inputs=[
                    "params:second_stage_industrial",
                    "best_second_stage_industrial_model",
                    "under_performers_industrial",
                    "top_performers_industrial",
                ],
                outputs=[
                    "second_stage_industrial_predictions",
                    "second_stage_industrial_shap_value",
                    "second_stage_industrial_customer_id",
                    "second_stage_industrial_feature_names",
                    "second_stage_industrial_explainer",
                    "second_stage_industrial_X",
                ],
                name="predict_second_stage_industrial",
            ),
        ]
    )  # type: ignore
