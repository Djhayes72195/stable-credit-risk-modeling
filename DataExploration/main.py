from DataProcessing.aggregator import Aggregator
from DataProcessing.pipeline import Pipeline
from .generate_feature_report import FeatureReport
from memory_profiler import profile
from .config import data_store
from .config import FEATURE_REPORT_PATH
from Utility.helper_functions import setup_logger
import psutil
import logging
import gc
from pathlib import Path
import polars as pl
import pandas as pd

def main():
    logger = setup_logger()
    logger.info("Beginning feature report process.")
    base_data = data_store.pop('df_base', None)

    base_df = Pipeline.read_file(base_data)

    unpack_and_iterate_through_tables(base_df, logger, data_store)


# def unpack_and_iterate_through_tables(base_df, logger, data_store):
#     numerical_metrics, categorical_metrics = pd.DataFrame(), pd.DataFrame()
#     feature_mapping = {'Source': [], 'Feature': []}
#     for depth in data_store.keys():
#         for path in data_store[depth]:
#             depth_num: int = int(depth.split("_")[-1])
#             multi_path = "*" in str(path)
#             if multi_path:
#                 path_to_scan = Path(str(path).replace("*", "0")) # All paths contained within * path share the same cols.
#                 feature_set = list(pl.scan_parquet(path_to_scan).columns)
#                 col_sets = split_features(feature_set, 4)
#             else:
#                 feature_set = list(pl.scan_parquet(path).columns)
#                 col_sets = [feature_set]
#             feature_mapping = update_feature_mapping(path, feature_set, feature_mapping)
#             for col_set in col_sets:
#                 metrics = generate_feature_report(base_df, logger, path, depth_num, col_set)
#                 numerical_metrics = pd.concat([numerical_metrics, metrics['numerical']], axis=0)
#                 categorical_metrics = pd.concat([categorical_metrics, metrics['categorical']], axis=0)
#                 del metrics
#                 gc.collect()
#     feature_mapping_df = pd.DataFrame(feature_mapping)
#     feature_mapping_df.to_csv(Path(FEATURE_REPORT_PATH / "FeatureMapping.csv"))
#     numerical_metrics.to_csv(Path(FEATURE_REPORT_PATH / "NumericalReport.csv"))
#     categorical_metrics.to_csv(Path(FEATURE_REPORT_PATH / "CategoricalReport.csv"))
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
    # for path in depth_2:
    #     col_sets = [None] # Will instruct pl to read all columns if not multi_path
    #     depth = 2
    #     multi_path = "*" in str(path)
    #     if multi_path:
    #         example_path = Path(str(path).replace("*", "0"))
    #         feature_set = list(pl.scan_parquet(example_path).columns)
    #         col_sets = split_features(feature_set, 4)
    #     for col_set in col_sets:
    #         metrics = generate_feature_report(base_df, logger, path, depth, col_set)
    #         numerical_metrics = pd.concat([numerical_metrics, metrics['numerical']], axis=0)
    #         categorical_metrics = pd.concat([categorical_metrics, metrics['categorical']], axis=0)
    #         del metrics
    #         gc.collect()
            
    # x = 2

def unpack_and_iterate_through_tables(base_df, logger, data_store):
    numerical_metrics, categorical_metrics = pd.DataFrame(), pd.DataFrame()
    feature_mapping = {'Source': [], 'Feature': []}
    
    for depth in data_store.keys():
        for path in data_store[depth]:
            depth_num = extract_depth(depth)
            feature_set, col_sets = handle_path_and_features(path)
            feature_mapping = update_feature_mapping(path, feature_set, feature_mapping)
            numerical_metrics, categorical_metrics = process_col_sets(
                base_df, logger, path, depth_num, col_sets, numerical_metrics, categorical_metrics
            )

    save_reports(feature_mapping, numerical_metrics, categorical_metrics)


def extract_depth(depth):
    return int(depth.split("_")[-1])


def handle_path_and_features(path):
    multi_path = "*" in str(path)
    if multi_path:
        path_to_scan = Path(str(path).replace("*", "0"))
        feature_set = list(pl.scan_parquet(path_to_scan).columns)
        col_sets = split_features(feature_set, 4)
    else:
        feature_set = list(pl.scan_parquet(path).columns)
        col_sets = [feature_set]
    return feature_set, col_sets


def process_col_sets(base_df, logger, path, depth_num, col_sets, numerical_metrics, categorical_metrics):
    for col_set in col_sets:
        metrics = generate_feature_report(base_df, logger, path, depth_num, col_set)
        numerical_metrics = pd.concat([numerical_metrics, metrics['numerical']], axis=0)
        categorical_metrics = pd.concat([categorical_metrics, metrics['categorical']], axis=0)
        del metrics
        gc.collect()
    return numerical_metrics, categorical_metrics


def save_reports(feature_mapping, numerical_metrics, categorical_metrics):
    feature_mapping_df = pd.DataFrame(feature_mapping)
    feature_mapping_df.to_csv(Path(FEATURE_REPORT_PATH / "FeatureMapping.csv"))
    numerical_metrics.to_csv(Path(FEATURE_REPORT_PATH / "NumericalReport.csv"))
    categorical_metrics.to_csv(Path(FEATURE_REPORT_PATH / "CategoricalReport.csv"))

def update_feature_mapping(path, features, dict_to_update):
    for feature in features:
        dict_to_update['Source'].append(str(path))
        dict_to_update['Feature'].append(str(feature))
    return dict_to_update

def split_features(features, n):
    features.remove('case_id')
    k, m = divmod(len(features), n)
    feature_split =  [features[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    for split in feature_split:
        split.append('case_id')
    return feature_split

@profile
def generate_feature_report(base_df, logger, path, depth, col_set):
    logger.info(f"Processing depth {depth} table.")
    read_function = Pipeline.read_files if "*" in str(path) else Pipeline.read_file
    # Track memory after reading the file
    train_df = read_function(path, col_set)
    logger.info(f"Memory after reading file: {psutil.Process().memory_info().rss / 1024 ** 2} MB")

    # Track memory after applying the pipeline
    train_df = train_df.pipe(Pipeline.apply_pipeline, base_df=base_df, depth=depth)
    logger.info(f"Memory after applying pipeline: {psutil.Process().memory_info().rss / 1024 ** 2} MB")

    feature_report = FeatureReport(train_df, logger)
    del train_df
    feature_report.generate_report()
    metrics = feature_report.metrics
    del feature_report
    gc.collect()
    return metrics


def my_function(read_function, base_df, depth, path):
    train_df = read_function(path).pipe(Pipeline.apply_pipeline, base_df=base_df, depth=depth)
    return train_df


if __name__ == '__main__':
    main()


