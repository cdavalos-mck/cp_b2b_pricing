from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_cluster_pipeline,
    get_k_nearest_neighbors,
    preprocess_columns,
    summarize_closest_customers,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_cluster_pipeline,
                inputs="params:clustering",
                outputs="cluster_pipeline",
                name="create_cluster_pipeline",
            ),
            node(
                func=preprocess_columns,
                inputs=[
                    "params:clustering",
                    "second_stage_predictions",
                    "cluster_pipeline",
                ],
                outputs=["X", "X_scaled"],
                name="preprocess_columns",
            ),
            node(
                func=get_k_nearest_neighbors,
                inputs=[
                    "params:clustering",
                    "second_stage_predictions",
                    "X_scaled",
                    "cluster_pipeline",
                ],
                outputs="k_nearest_neighbors",
                name="get_k_nearest_neighbors",
            ),
            node(
                func=summarize_closest_customers,
                inputs=[
                    "params:clustering",
                    "second_stage_predictions",
                    "k_nearest_neighbors",
                    "X_scaled",
                ],
                outputs=["explanation", "closest_customers_detail"],
                name="summarize_closest_customers",
            ),
        ]
    )  # type: ignore
