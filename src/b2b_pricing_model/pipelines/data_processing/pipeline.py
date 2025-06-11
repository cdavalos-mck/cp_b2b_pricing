from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_columns,
                inputs="raw_base_cupon_electronico",
                outputs="int_base_cupon_electronico",
                name="preprocess_base_cupon_electronico",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_base_tct_tae",
                outputs="int_base_tct_tae",
                name="preprocess_base_tct_tae",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_base_empresas",
                outputs="int_base_empresas",
                name="preprocess_base_empresas",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_contratos_tct",
                outputs="int_contratos_tct",
                name="preprocess_contratos_tct",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_contratos_tae",
                outputs="int_contratos_tae",
                name="preprocess_contratos_tae",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_additional_contratos_tae",
                outputs="int_additional_contratos_tae",
                name="preprocess_additional_contratos_tae",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_competitiveness_index",
                outputs="int_competitiveness_index",
                name="preprocess_competitiveness_index",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_maestro_data",
                outputs="int_maestro_data",
                name="preprocess_maestro_data",
            ),
            node(
                func=preprocess_columns,
                inputs="raw_patentes",
                outputs="int_patentes",
                name="preprocess_patentes",
            ),
        ]
    )  # type: ignore
