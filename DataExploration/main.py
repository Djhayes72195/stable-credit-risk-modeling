from DataProcessing.aggregator import Aggregator
from DataProcessing.pipeline import Pipeline
from .generate_feature_report import FeatureReport
from memory_profiler import profile
from .config import data_store
from Utility.helper_functions import setup_logger
import psutil
import logging
import polars as pl
import pandas as pd

def main():
    logger = setup_logger()
    logger.info("Beginning feature report process.")
    base_data = data_store.pop('df_base', None)

    base_df = Pipeline.read_file(base_data)

    unpack_and_iterate_through_tables(base_df, logger, **data_store)


def unpack_and_iterate_through_tables(base_df, logger, depth_0, depth_1, depth_2):
    numerical_metrics, categorical_metrics = pd.DataFrame(), pd.DataFrame()
    # TODO: Make DRY
    # for path in depth_0:
    #     depth = 0
    #     metrics = generate_feature_report(base_df, logger, path, depth)
    #     numerical_metrics = pd.concat([numerical_metrics, metrics['numerical']], axis=0)
    #     categorical_metrics = pd.concat([categorical_metrics, metrics['categorical']], axis=0)
    # for path in depth_1:
    #     depth = 1
    #     metrics = generate_feature_report(base_df, logger, path, depth)
    #     numerical_metrics = pd.concat([numerical_metrics, metrics['numerical']], axis=0)
    #     categorical_metrics = pd.concat([categorical_metrics, metrics['categorical']], axis=0)
    for path in depth_2:
        depth = 2
        metrics = generate_feature_report(base_df, logger, path, depth)
        numerical_metrics = pd.concat([numerical_metrics, metrics['numerical']], axis=0)
        categorical_metrics = pd.concat([categorical_metrics, metrics['categorical']], axis=0)
    x = 2

@profile
def generate_feature_report(base_df, logger, path, depth):
    logger.info(f"Processing depth {depth} table.")
    read_function = Pipeline.read_files if "*" in str(path) else Pipeline.read_file
    # Track memory after reading the file
    train_df = read_function(path)
    logger.info(f"Memory after reading file: {psutil.Process().memory_info().rss / 1024 ** 2} MB")

    # Track memory after applying the pipeline
    train_df = train_df.pipe(Pipeline.apply_pipeline, base_df=base_df, depth=depth)
    logger.info(f"Memory after applying pipeline: {psutil.Process().memory_info().rss / 1024 ** 2} MB")

    feature_report = FeatureReport(train_df, logger)
    feature_report.generate_report()
    return feature_report.metrics


def my_function(read_function, base_df, depth, path):
    train_df = read_function(path).pipe(Pipeline.apply_pipeline, base_df=base_df, depth=depth)
    return train_df


if __name__ == '__main__':
    main()


