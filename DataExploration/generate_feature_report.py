"""
A Utility class used to generate a report detailing simple univariate and bi-variate statistics on features.

The purpose this class is calculate basic statistics on for features and organize them into
a report. Each row of the report corresponds to a feature and each column a statistic or metric.

The following statistics will be included:

- Mean
- Median
- Variance
- Weight of Evidence
- Information Value
- Correlation with Target
- Number of missing values
- Data type (categorical or numerical)
- Number of unique values (categorical)
- Min and max values (numerical)
- Skewedness
- Kurtosis

Each statistic will be generated as a particular method.
"""


from .config import FeaturePartitionEnum, BUCKET_NAME, CORE_COLUMNS
import pandas as pd
import polars as pl
import boto3
import numpy as np
from io import BytesIO
from DataProcessing.pipeline import Pipeline

CHUNK_SIZE = 10
POLARS_NUMERIC_TYPES = (pl.Int32, pl.Int64, pl.Float32, pl.Float64)

class FeatureReport:

    def __init__(self, base_df, train_df):
        self.base_df = base_df
        self.columns_to_report = train_df.drop('case_id').columns
        self.numerical_cols = []
        self.cat_cols = []

        # Join base df because it contains columns required for preprocessing
        self.train_df = base_df.join(train_df, how="left", on="case_id").pipe(Pipeline.handle_dates)

        for col in self.columns_to_report:
            if self.train_df.dtypes[self.train_df.columns.index(str(col))] in POLARS_NUMERIC_TYPES:
                self.numerical_cols.append(col)
            else:
                self.cat_cols.append(col)

        self.numeric_report = pd.DataFrame()
        self.catagorical_report = pd.DataFrame()
        self.means = pd.DataFrame()
        self.variances = pd.DataFrame()
        self.medians = pd.DataFrame()

    def generate_report(self):
        self.calculate_mean()
        self.calculate_median()
        self.calculate_variance()
        self.calculate_woe()
        pass
    
    def calculate_mean(self):
        """
        Return a dict which maps the name of the col to the mean
        value of the column.
        """
        # cols_to_report = train_df.columns
        for col in self.numerical_cols:
            mean = self.train_df.select(pl.col(col).mean()).to_numpy()[0]
            self.means[col] = mean
        else:
            self.means[col] = np.nan

        # train_df = self.base_df.join(train_df, how="left", on="case_id").pipe(Pipeline.handle_dates)
        # feature_name_to_mean_dict = {
        #     str(col): train_df.select(pl.col(col).mean()).to_numpy()[0]
        #     for col in train_df.columns
        #     if train_df[col].dtype in POLARS_NUMERIC_TYPES
        #     and col in cols_to_report
        # }

    def calculate_median(self):
        for col in self.numerical_cols:
            median = self.train_df.select(pl.col(col).median()).to_numpy()[0]
            self.medians[col] = median
        else:
            self.medians[col] = np.nan     

    def calculate_variance(self):
        for col in self.numerical_cols:
            var = self.train_df.select(pl.col(col).var()).to_numpy()[0]
            self.variances[col] = var
        else:
            self.variances[col] = np.nan

    def calculate_woe(self):
        woe_dict = {}
        for col in self.cat_cols:
            df = self.train_df.filter(pl.col(col).is_not_null())
            carinality = df.select(pl.col(col).n_unique()).to_numpy()[0]
            if carinality == 1 or carinality > 100:
                print(f"{col} has unsuitable cardinality {carinality}, skipping woe calculation.")
                continue
            total_goods = df.filter(pl.col('target') == 0).count()
            total_bads = df.filter(pl.col('target') == 1).count()

            category_goods =df.groupby(col).agg([
                                        (pl.col("target") == 0).sum().alias("goods")
                                    ])
            category_bads = df.groupby(col).agg([
                (pl.col('target') == 1).sum().alias('bads')
            ])

            # Calculate WoE for each category
            # WoE = ln((goods / total_goods) / (bads / total_bads))
            woe_df = category_goods.join(category_bads, on=col, how='outer').with_columns(
                (pl.col('goods') / total_goods / (pl.col('bads') / total_bads)).log().alias('WoE')
            )

            pass
# if __name__ == '__main__':
#     s3 = boto3.client("s3")
#         # Load base table
#     object_key = "train/base/train_base.csv"
#     base_feature_path = 'train/feature'

#     reponse = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)

#     content = reponse['Body'].read()
#     base_df = pd.read_csv(BytesIO(content))

#     for feature_class in FeaturePartitionEnum:
#         if feature_class.value == 'applprev':
#             feature_class_partition = base_feature_path + '/' + feature_class.value
#             report = FeatureReport(base_df, feature_class_partition, s3)
#             report.combine_tables()
    