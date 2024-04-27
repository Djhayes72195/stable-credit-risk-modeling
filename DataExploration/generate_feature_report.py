"""
A Utility class used to generate a report detailing simple univariate and bi-variate statistics on features.

The purpose this class is calculate basic statistics on for features and organize them into
a report. Each row of the report corresponds to a feature and each column a statistic or metric.

The following statistics will be included:

- Mean
- Median
- Standard Deviation
- Weight of Evidence
- Information Value
- Correlation with Target
- Number of missing values
- Data type (categorical or numerical)
- Number of unique values (categorical)
- Min and max values (numberical)
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

    def __init__(self, base_df):
        self.base_df = base_df

    
    # def combine_tables(self):
    #     response = self.s3_client.list_objects_v2(
    #         Bucket=BUCKET_NAME,
    #         Prefix=self.feature_class_partition
    #     )        
    #     if 'Contents' in response:
    #         for obj in response['Contents']:
    #             # Construct the file key
    #             file_key = obj['Key']
                
    #             # Skip directories
    #             if file_key.endswith('/'):
    #                 continue
                
    #             # Get the object (file) using the key
    #             file_obj = self.s3_client.get_object(
    #                 Bucket=BUCKET_NAME,
    #                 Key=file_key
    #             )
    #             for chunk in pd.read_csv(BytesIO(file_obj['Body'].read()), chunksize=CHUNK_SIZE):
    #                 print(chunk)
    
    def calculate_mean(self, train_df):
        """
        Return a dict which maps the name of the col to the mean
        value of the column.
        """
        cols_to_report = train_df.columns
        train_df = self.base_df.join(train_df, how="left", on="case_id").pipe(Pipeline.handle_dates)
        feature_name_to_mean_dict = {
            str(col): train_df.select(pl.col(col).mean()).to_numpy()[0]
            for col in train_df.columns
            if train_df[col].dtype in POLARS_NUMERIC_TYPES
            and col in cols_to_report
        }
        return feature_name_to_mean_dict    
            

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
    