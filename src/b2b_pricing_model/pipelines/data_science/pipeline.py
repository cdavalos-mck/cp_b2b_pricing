from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_master_base, get_best_hyperparameters


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_master_base,
                inputs=["params:outliers", "master_trx_customer_tct"],
                outputs=[
                    "non_regular_clients_tct",
                    "regular_base_clients_tct",
                    "regular_outlier_clients_tct",
                ],
                name="create_master_base_tct",
            ),
            node(
                func=get_best_hyperparameters,
                inputs=[
                    "params:first_stage",
                    "regular_base_clients_tct",
                ],
                outputs="first_stage_best_hyperparameters",
                name="get_first_stage_best_hyperparameters",
            ),
        ]
    )  # type: ignore
