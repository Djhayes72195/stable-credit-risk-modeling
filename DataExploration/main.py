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
    feature_report = FeatureReport(base_df)
    for path in depth_0:
        if "*" in str(path):
            print("Do nothing for now. * represents multiple related tables.")
        else:
            train_df = Pipeline.read_file(path)
            # train_df = base_df.join(train_df, how="left", on="case_id").pipe(Pipeline.handle_dates)
            # cols_to_report = train_df.columns
            means_report = feature_report.calculate_mean(train_df)

if __name__ == '__main__':
    main()


