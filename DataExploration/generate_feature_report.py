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

Each statistic will be generated with a particular method.

The feature report should take this general form:

Numeric report:
            | mean_no_default | mean_default | median_no_default | median_default |   .......
feature 1   |      val        |     val      |     .........     |     ........   |
feature 2   |
    ...
    ...                                     ......


Categorical report:
            | WoE_feature_class1 | WoE_feature_class2 |       ...      | Information Value |   .......
feature 1   |      val           |     val            |     .........  |     ........      |
feature 2   |
    ...
    ...                                     ......

The ultimate destination of this data is a set of dashboards generated with plotly.
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
DEFAULT = "_default"
NO_DEFAULT = "_no_default"

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
        mean_values = {}
        for col in self.numerical_cols:
            try:
                mean = self.train_df.select(pl.col(col).mean()).to_numpy()[0]
                mean_no_default = self.train_df.filter(pl.col('target') == 0).select(pl.col(col).mean()).to_numpy()[0]
                mean_default = self.train_df.filter(pl.col('target') == 1).select(pl.col(col).mean()).to_numpy()[0]
                mean_values[col] = mean
                mean_values[col + NO_DEFAULT] = mean_no_default
                mean_values[col + DEFAULT] = mean_default
            except Exception as e:
                print(f"Error calculating mean for column {col}: {str(e)}")
                mean_values[col] = np.nan
        for col in self.cat_cols:
            self.means[col] = np.nan
        
        self.means = pd.DataFrame.from_dict(mean_values, orient='index', columns=['mean'])


    def calculate_median(self):
        median_values = {}
        for col in self.numerical_cols:
            try:
                median = self.train_df.select(pl.col(col).median()).to_numpy()[0]
                median_values[col] = median
            except Exception as e:
                print(f"Error calculating median for column {col}: {str(e)}")
                median_values[col] = median

        for col in self.cat_cols:
            self.median[col] = np.nan

        self.medians = pd.DataFrame.from_dict(median_values, orient='index', columns=['median'])    

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
    