"""
A Utility class used to generate a report detailing simple univariate and bi-variate statistics on features.

The purpose this class is calculate basis statics on our features and organize them into
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


from config import FeaturePartitionEnum
import pandas as pd
import polars as pl
import boto3
from config import BUCKET_NAME
from io import BytesIO

CHUNK_SIZE = 10

class FeatureReport:

    def __init__(self, base_df, feature_class_partition, s3_client):
        self.base_df = base_df
        self.feature_class_partition = feature_class_partition
        self.s3_client = s3_client
    
    def combine_tables(self):
        response = self.s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=self.feature_class_partition
        )        
        if 'Contents' in response:
            for obj in response['Contents']:
                # Construct the file key
                file_key = obj['Key']
                
                # Skip directories
                if file_key.endswith('/'):
                    continue
                
                # Get the object (file) using the key
                file_obj = self.s3_client.get_object(
                    Bucket=BUCKET_NAME,
                    Key=file_key
                )
                for chunk in pd.read_csv(BytesIO(file_obj['Body'].read()), chunksize=CHUNK_SIZE):
                    print(chunk)

if __name__ == '__main__':
    s3 = boto3.client("s3")
        # Load base table
    object_key = "train/base/train_base.csv"
    base_feature_path = 'train/feature'

    reponse = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)

    content = reponse['Body'].read()
    base_df = pd.read_csv(BytesIO(content))

    for feature_class in FeaturePartitionEnum:
        if feature_class.value == 'applprev':
            feature_class_partition = base_feature_path + '/' + feature_class.value
            report = FeatureReport(base_df, feature_class_partition, s3)
            report.combine_tables()
    