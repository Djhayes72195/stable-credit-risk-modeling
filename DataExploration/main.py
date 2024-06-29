from DataProcessing.aggregator import Aggregator
from DataProcessing.pipeline import Pipeline
from .generate_feature_report import FeatureReport
from .config import data_store
import polars as pl

def main():
    base_data = data_store.pop('df_base', None)
    print('It did run...')

    base_df = Pipeline.read_file(base_data)

    unpack_and_iterate_through_tables(base_df, **data_store)


def unpack_and_iterate_through_tables(base_df, depth_0, depth_1, depth_2):
    for path in depth_0:
        read_function = Pipeline.read_files if "*" in str(path) else Pipeline.read_file
        train_df = read_function(path)
        feature_report = FeatureReport(base_df, train_df)
        feature_report.generate_report()


if __name__ == '__main__':
    main()


