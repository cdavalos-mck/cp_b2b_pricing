"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from b2b_pricing_model.pipelines import data_processing, data_science


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = (
        data_processing.create_pipeline()
        + data_science.create_pipeline()
        + data_science.create_segments_pipeline()
    )
    pipelines["data_engineering"] = data_processing.create_pipeline()
    pipelines["data_science"] = data_science.create_pipeline()
    pipelines["pricing"] = data_science.create_pricing_pipeline()
    pipelines["segments"] = data_science.create_segments_pipeline()
    pipelines["pricing_industrial"] = data_science.create_pricing_industrial_pipeline()
    return pipelines
