from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_cluster_pipeline, get_k_nearest_neighbors, preprocess_columns


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
                    "master_trx_customer_tct",
                    "cluster_pipeline",
                ],
                outputs=["X", "X_scaled"],
                name="preprocess_columns",
            ),
            node(
                func=get_k_nearest_neighbors,
                inputs=[
                    "params:clustering",
                    "master_trx_customer_tct",
                    "X_scaled",
                    "cluster_pipeline",
                ],
                outputs="k_nearest_neighbors",
                name="get_k_nearest_neighbors",
            ),
        ]
    )  # type: ignore
