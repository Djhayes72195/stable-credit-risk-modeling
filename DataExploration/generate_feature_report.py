"""
A Utility class used to generate a report detailing simple univariate and bi-variate statistics on features.

The purpose this class is calculate basic statistics on features and organize them into
a report. Each row of the report corresponds to a feature and each column a statistic or metric.

The following statistics will be included:

- Mean (default/no_default)
- Median (default/no_default)
- Variance (default/no_default)
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
        self.woes = pd.DataFrame()

    def generate_report(self):
        # self.calculate_mean()
        # self.calculate_median()
        # self.calculate_variance()
        self.calculate_woe()
        pass
    
    def calculate_mean(self):
        mean_values = {
            'mean': [],
            'mean_no_default': [],
            'mean_default': []
        }
        feature = []
        for col in self.numerical_cols:
            try:
                mean = self.train_df.select(pl.col(col).mean())[0, 0]
                mean_no_default = self.train_df.filter(pl.col('target') == 0).select(pl.col(col).mean())[0, 0]
                mean_default = self.train_df.filter(pl.col('target') == 1).select(pl.col(col).mean())[0, 0]
                feature.append(col)
                mean_values['mean'].append(mean)
                mean_values['mean_no_default'].append(mean_no_default)
                mean_values['mean_default'].append(mean_default)
            except Exception as e:
                print(f"Error calculating mean for column {col}: {str(e)}")
                feature.append(col)
                mean_values['mean'].append(np.nan)
                mean_values['mean_no_default'].append(np.nan)
                mean_values['mean_default'].append(np.nan)

        for col in self.cat_cols:
            feature.append(col)
            mean_values['mean'].append(np.nan)
            mean_values['mean_no_default'].append(np.nan)
            mean_values['mean_default'].append(np.nan)

        self.means = pd.DataFrame(mean_values, index=feature)

        
        # self.means = pd.DataFrame.from_dict(mean_values, orient='index', columns=['mean'])


    def calculate_median(self):
        median_values = {
            'median': [],
            'median_no_default': [],
            'median_default': []
        }
        feature = []
        
        for col in self.numerical_cols:
            try:
                median = self.train_df.select(pl.col(col).median())[0, 0]
                median_no_default = self.train_df.filter(pl.col('target') == 0).select(pl.col(col).median())[0, 0]
                median_default = self.train_df.filter(pl.col('target') == 1).select(pl.col(col).median())[0, 0]
                feature.append(col)
                median_values['median'].append(median)
                median_values['median_no_default'].append(median_no_default)
                median_values['median_default'].append(median_default)
            except Exception as e:
                print(f"Error calculating median for column {col}: {str(e)}")
                feature.append(col)
                median_values['median'].append(np.nan)
                median_values['median_no_default'].append(np.nan)
                median_values['median_default'].append(np.nan)

        for col in self.cat_cols:
            feature.append(col)
            median_values['median'].append(np.nan)
            median_values['median_no_default'].append(np.nan)
            median_values['median_default'].append(np.nan)

        self.medians = pd.DataFrame(median_values, index=feature) 

    def calculate_variance(self):
        var_values = {
            'variance': [],
            'variance_no_default': [],
            'variance_default': []
        }
        feature = []
        
        for col in self.numerical_cols:
            try:
                var = self.train_df.select(pl.col(col).var())[0, 0]
                var_no_default = self.train_df.filter(pl.col('target') == 0).select(pl.col(col).var())[0, 0]
                var_default = self.train_df.filter(pl.col('target') == 1).select(pl.col(col).var())[0, 0]
                feature.append(col)
                var_values['variance'].append(var)
                var_values['variance_no_default'].append(var_no_default)
                var_values['variance_default'].append(var_default)
            except Exception as e:
                print(f"Error calculating variance for column {col}: {str(e)}")
                feature.append(col)
                var_values['variance'].append(np.nan)
                var_values['variance_no_default'].append(np.nan)
                var_values['variance_default'].append(np.nan)

        for col in self.cat_cols:
            feature.append(col)
            var_values['variance'].append(np.nan)
            var_values['variance_no_default'].append(np.nan)
            var_values['variance_default'].append(np.nan)

        self.variances = pd.DataFrame(var_values, index=feature)

    def calculate_woe(self):
        woe_dict = {}
        for col in self.cat_cols:
            df = self.train_df.select(pl.col(col), pl.col('target')).filter(pl.col(col).is_not_null())
            cardinality = df.select(pl.col(col).n_unique())[0, 0]
            if cardinality == 1 or cardinality > 100:
                print(f"{col} has unsuitable cardinality {cardinality}, skipping woe calculation.")
                continue
            
            total_goods = df.filter(pl.col('target') == 0).shape[0]
            total_bads = df.filter(pl.col('target') == 1).shape[0]

            category_stats = df.groupby(col).agg([
                (pl.col("target") == 0).sum().alias("goods"),
                (pl.col("target") == 1).sum().alias("bads")
            ])

            # Calculate WoE for each category
            # WoE = ln((goods / total_goods) / (bads / total_bads))
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-10
            category_stats = category_stats.with_columns([
                ((pl.col('goods') / pl.lit(total_goods)) / (pl.col('bads') / pl.lit(total_bads) + pl.lit(epsilon))).log().alias('WoE')
            ])

            woe_dict[col] = category_stats.select(pl.col(col), pl.col('WoE'))

        # Convert the woe_dict to a DataFrame if needed
        # You might want to concatenate them into a single DataFrame depending on your needs
        self.woe_df = pd.concat([df.to_pandas() for df in woe_dict.values()], ignore_index=True)


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
    