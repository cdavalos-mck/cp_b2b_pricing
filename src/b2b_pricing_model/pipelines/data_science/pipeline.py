from kedro.pipeline import Pipeline

from b2b_pricing_model.pipelines.data_science.customer_segments import (
    pipeline as segments_pipeline,
)
from b2b_pricing_model.pipelines.data_science.pricing_model import (
    pipeline as pricing_pipeline,
)


def create_pipeline(**kwargs) -> Pipeline:
    ds_pipe = pricing_pipeline.create_pipeline() + segments_pipeline.create_pipeline()
    return ds_pipe


def create_pricing_pipeline(**kwargs) -> Pipeline:
    return pricing_pipeline.create_pipeline()


def create_segments_pipeline(**kwargs) -> Pipeline:
    return segments_pipeline.create_pipeline()
