"""
A Utility class used to generate a report detailing simple univariate and bi-variate statistics on features.

The purpose this class is calculate basic statistics on features and organize them into
a report. Each row of the report corresponds to a feature and each column a statistic or metric.

The following statistics will be included:

- Mean (numerical, default/no_default)
- Median (numerical, default/no_default)
- Variance (numerical, default/no_default)
- Weight of Evidence (categorical, don't display, store for analysis)
- Information Value (categorical)
- Correlation with Target (single)
- Number of missing values (numerical or categorical)
- Number of unique values (categorical)
- Chi Squared test (categorical)
- Min and max values (numerical)
- Skewedness
- Kurtosis

If time permits:
- Information gain (categorical)
- Fisher value 

Each statistic will be generated with a particular method.

The feature report should take this general form:

Numeric report:
            | mean_no_default | mean_default | median_no_default | median_default |   .......
feature 1   |      val        |     val      |     .........     |     ........   |
feature 2   |
    ...
    ...                                     ......


Categorical report:
            |         WoE        | Information Value |   .......      |                   |
feature 1   |         df         |        val        |     .........  |     ........      |
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
from collections import Counter
from DataProcessing.pipeline import Pipeline
from scipy.stats import chi2_contingency, skew, kurtosis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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
        self.IVs = pd.DataFrame()
        self.num_unique = pd.DataFrame()
        self.pct_na_vals_cat = pd.DataFrame() # separate number of unique values into numerical and categorical
        self.pct_na_vals_num = pd.DataFrame()
        self.chi2 = pd.DataFrame()
        self.min_and_max = pd.DataFrame()
        self.skew = pd.DataFrame()
        self.kurtosis = pd.DataFrame()

    def generate_report(self):
        # self.calculate_mean()
        # self.calculate_median()
        # self.calculate_min_and_max()
        # self.calculate_variance()
        # self.calculate_pct_na()
        # self.calculate_chi_squ_test()
        self.calculate_mutual_info()
        self.calculate_skew()
        self.calculate_kurtosis()
        # self.calculate_woe_iv()
        # self.calculate_correlation_with_target()
        self.calculate_number_of_unique_val()
        pass

    def calculate_mutual_info(self):
        """
        Measure mutual information
        
        Mutual information is a measurement of the degree of independence of
        two variables. If the variables are completely independent, we expect
        a mutual information value of around 0. If they are highly dependent, we
        expect a mutual information value of close to 1.

        Columns with more than .8 nullness are removed before this calculation.
        Remaining nulls are imputed using sklearn SimpleImputer. Nulls in numerical
        columns are filled with the mean of that column. Nulls in categorical columns
        are filled with the mode.
        """
        mi_cat = {}
        mi_num = {}
        target = self.train_df['target']

        # Ensure pct of na and num of unique vals is calculated first
        if self.pct_na_vals_cat.empty or self.pct_na_vals_num.empty:
            self.calculate_pct_na()
        pct_na = pd.concat([self.pct_na_vals_cat, self.pct_na_vals_num], ignore_index=False)

        if self.num_unique.empty:
            self.calculate_number_of_unique_val()
        
        cols_to_drop_cardinality = set(self.num_unique[self.num_unique['NumUnique'] > 20])
        cols_to_drop_na = set(pct_na[pct_na['PctNA'] > .8].index)
        cols_to_drop = list(cols_to_drop_cardinality | cols_to_drop_na)
        cols = [col for col in self.columns_to_report if col not in cols_to_drop]

        df = self.train_df[cols].to_pandas()

        # Impute values for NaN
        # Impute missing values for numeric features
        imputer_num = SimpleImputer(strategy='mean')
        df[df.select_dtypes(include=[np.number]).columns] = imputer_num.fit_transform(df.select_dtypes(include=[np.number]))

        # Impute missing values for categorical features with most frequent value
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[df.select_dtypes(include=['object']).columns] = imputer_cat.fit_transform(df.select_dtypes(include=['object']))


        # Encode categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        mi = mutual_info_classif(df, target)



    def calculate_skew(self):
        skew_dict = {}
        for col in self.numerical_cols:
            skewedness = skew(self.train_df[col].drop_nulls())
            skew_dict[col] = skewedness
        self.skew = pd.DataFrame.from_dict(skew_dict, orient='index', columns=['Skewedness'])

    def calculate_kurtosis(self):
        kurt_dict = {}
        for col in self.numerical_cols:
            kurt = kurtosis(self.train_df[col].drop_nulls())
            kurt_dict[col] = kurt
        self.skew = pd.DataFrame.from_dict(kurt_dict, orient='index', columns=['Skewedness'])       

    def calculate_pct_na(self):
        pct_na_cat_dict = {}
        pct_na_num_dict = {}
        total_num_of_val = self.train_df.shape[0]
        for col in self.columns_to_report:
            num_na = self.train_df[col].is_null().sum()
            if col in self.cat_cols:
                pct_na_cat_dict[col] = num_na / total_num_of_val
            else:
                pct_na_num_dict[col] = num_na / total_num_of_val
        self.pct_na_vals_cat = pd.DataFrame.from_dict(pct_na_cat_dict, orient='index', columns=['PctNA'])
        self.pct_na_vals_num = pd.DataFrame.from_dict(pct_na_num_dict, orient='index', columns=['PctNA'])

    def calculate_chi_squ_test(self):
        chi_squ_test_res = {
            "Significant": [],
            "P-val": [],
            "Chi2": []
        }
        feature = []
        for col in self.cat_cols:
            feature.append(col)
            df = self.train_df.select(pl.col(col), pl.col('target')).to_pandas()
            df_contingency = pd.crosstab(df[col], df['target'])
            chi2, p, _, _ = chi2_contingency(df_contingency)
            chi_squ_test_res["Significant"].append("yes" if p < .05 else "no")
            chi_squ_test_res['P-val'].append(p)
            chi_squ_test_res['Chi2'].append(chi2)
        self.chi2 = pd.DataFrame(chi_squ_test_res, index=feature)

    def calculate_min_and_max(self):
        min_and_max = {
            "min": [],
            "max": []
        }
        feature = []
        for col in self.numerical_cols:
            min_and_max['min'].append(self.train_df[col].min())
            min_and_max['max'].append(self.train_df[col].max())
            feature.append(col)
        self.min_and_max = pd.DataFrame(min_and_max, index=feature)
    
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

        # for col in self.cat_cols:
        #     feature.append(col)
        #     mean_values['mean'].append(np.nan)
        #     mean_values['mean_no_default'].append(np.nan)
        #     mean_values['mean_default'].append(np.nan)

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

        # for col in self.cat_cols:
        #     feature.append(col)
        #     median_values['median'].append(np.nan)
        #     median_values['median_no_default'].append(np.nan)
        #     median_values['median_default'].append(np.nan)

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

        # for col in self.cat_cols:
        #     feature.append(col)
        #     var_values['variance'].append(np.nan)
        #     var_values['variance_no_default'].append(np.nan)
        #     var_values['variance_default'].append(np.nan)

        self.variances = pd.DataFrame(var_values, index=feature)

    def calculate_correlation_with_target(self):
        # implement point_biserial_corr
        pass

    def calculate_number_of_unique_val(self):
        num_unique_dict = {}
        for col in self.cat_cols:
            num_unique = self.train_df.select(pl.col(col).n_unique())[0, 0]
            num_unique_dict[col] = num_unique
        # for col in self.numerical_cols:
        #     num_unique_dict[col] = np.nan
        self.num_unique = pd.DataFrame.from_dict(num_unique_dict, orient='index', columns=['NumUnique'])


    def calculate_woe_iv(self):
        """
        WoE = ln(prop_goods/prop_bads)
        IV = sum((prop_goods - prop_bads) * WoE)

        WoE is calculated for each feature / category combination
        IV is calculated for each feature.

        WoE df is flattened w/ columns [feature, category, WoE]

        TODO: Review calc and set numerical cols to nan.
        """
        woe_dict = {}
        iv_dict = {}
        features = []
        for col in self.cat_cols:
            df = self.train_df.select(pl.col(col), pl.col('target')).filter(pl.col(col).is_not_null())
            cardinality = df.select(pl.col(col).n_unique())[0, 0]
            if cardinality == 1 or cardinality > 100:
                print(f"{col} has unsuitable cardinality {cardinality}, skipping woe calculation.")
                continue
            features.append(col)
            
            total_goods = df.filter(pl.col('target') == 0).shape[0]
            total_bads = df.filter(pl.col('target') == 1).shape[0]

            category_stats = df.groupby(col).agg([
                (pl.col("target") == 0).sum().alias("goods"),
                (pl.col("target") == 1).sum().alias("bads")
            ])

            # Add a small epsilon to avoid division by zero
            epsilon = 1e-10
            category_stats = category_stats.with_columns([
                (pl.col('goods') / (pl.lit(total_goods) + epsilon)).alias('prop_goods'),
                (pl.col('bads') / (pl.lit(total_bads) + epsilon)).alias('prop_bads')
            ])
            category_stats = category_stats.with_columns([
                (pl.col('prop_goods') / pl.col('prop_bads') + epsilon).log().alias('WoE')
            ])
            category_stats = category_stats.with_columns([
                ((pl.col('prop_goods') - pl.col('prop_bads')) * pl.col('WoE')).alias('IV')
            ])
            woe_dict[col] = category_stats.select(pl.col(col), pl.col('WoE')).to_pandas()
            iv = category_stats.select(pl.col('IV').sum()).item()
            iv_dict[col] = iv

        # flatten dictionary
        woe_list = []
        for feature, df in woe_dict.items():
            df['feature'] = feature
            df = df.rename(columns={feature: 'category'})
            woe_list.append(df)

        woe_df = pd.concat(woe_list, ignore_index=True)

        # Reorder columns to have 'feature' as the first column
        self.woes = woe_df[['feature', 'category', 'WoE']]
        self.IVs = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV'])

        


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
    