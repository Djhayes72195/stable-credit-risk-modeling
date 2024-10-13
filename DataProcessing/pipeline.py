
import polars as pl
import numpy as np
from .aggregator import Aggregator
from glob import glob


class Pipeline:

    @staticmethod
    def apply_pipeline(df, base_df, depth=None, drop_base_date_cols=True):
        """
        Apply all relevant pipeline functions to a dataframe.

        Parameters:
        - df: The DataFrame to process.
        - depth: The depth for aggregation (if applicable).
        - drop_base_date_cols: Whether to drop base date columns after processing dates.
            - Currently unused, included for flexibility
        Returns:
        - Processed DataFrame.
        """
        df = df.join(base_df, on='case_id')
        df = Pipeline.set_table_dtypes(df)
        df = Pipeline.handle_dates(df) # Drops MONTH and descision date columns from base_df
        df = Pipeline.filter_cols(df)
        df = Pipeline.reduce_mem_usage_polars(df)

        if depth in [1, 2]:
            target = df[['target', 'case_id']] # ensure target is not affected by aggregation.
            df = df.drop(columns=['target'])
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
            df = df.join(target, on='case_id')
        return df

    def set_table_dtypes(df):
        """
        Set data types.

        Used in the data loading pipeline to set data types
        for efficiency.

        "case_id", "WEEK_NUM", "num_group1", "num_group2" are special
        columns. case_id is the identifier for each case, WEEK_NUM indicates
        the week that an observation was taken, and num_group1 and num_group2
        are indexes used in depth=1 and depth=2 tables. Int is suitable for each.

        date_decision is likewise special: it denotes the date that the choice
        to either deny or issue the loan was made.

        Datatypes for remaining columns are selected based on the last character
        in the column name:

        P - Transform DPD (Days past due) - Float
        M - Masking categories - String
        A - Transform amount - Float
        D - Transform date - Date
        T - Unspecified Transform - Not handled
        L - Unspecified Transform - Not handled
        """
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df, drop_base_date_cols=True):
        """
        Convert date values to a # days difference from date decision.

        This function locates date columns (ending in D) and converts
        to a # of days difference between the date decision and the date of interest.

        Note: This function requires the "date_decision" column from the base_df.
        """
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  # Duration
                df = df.with_columns(pl.col(col).dt.total_days()) # Count of days, Polars handles conversion to int automatically
        if drop_base_date_cols:
            df = df.drop("date_decision", "MONTH", "WEEK_NUM") # Drop time data originating from base_df
        return df

    def filter_cols(df):
        """
        Filters columns based on # of null values and frequency of categorical values,
        with exceptions for specific columns.

        Logic:
            - Drop if greater than 70% of the column is null and not in "target", "case_id", "WEEK_NUM"
            - If categorical and not in "target", "case_id", "WEEK_NUM":
                - Drop if column has only one unique value (not informative)
                - Drop if column has more than 200 unique values (high cardinality, expensive and
                may lead to overfitting)
        """
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        
        return df
    
    def reduce_mem_usage(df):
        """
        Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.

        This function operates under the principle that a numerical column
        should use the more memory efficient datatype possible. For example,
        if the largest and smallest int in a column can be represented with int8, we should
        use int8 instead of a of larger type such as int16. It performs similar operations
        on floats such that the most memory efficient datatype is employed.

        "category" and "object" types are skipped as they are not suitable for this
        sort of type casting.

        The memory usage of the dataframe is recorded and printed before and after
        the operation, allowing the user to know the extent to which memory usage was
        optimized.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in df.columns:
            col_type = df[col].dtype
            if str(col_type)=="category":
                continue
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                continue
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
        return df

    def reduce_mem_usage_polars(df):
        """
        Iterate through all the columns of a Polars DataFrame and modify the data type
        to reduce memory usage.
        """
        start_mem = df.estimated_size() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            # Skip categorical and object-like columns
            if col_type in [pl.Categorical, pl.Utf8]:
                continue

            c_min = df[col].min()
            c_max = df[col].max()

            if col_type in [pl.Int64, pl.Int32]:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(pl.col(col).cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(pl.col(col).cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif col_type == pl.Float64:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(pl.col(col).cast(pl.Float32))

        end_mem = df.estimated_size() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df


    def read_file(path, col_set=None):
        """
        Reads a parquet file
        """
        df = pl.read_parquet(path, columns=col_set)
        # df = df.pipe(Pipeline.set_table_dtypes)
        # if depth in [1,2]:
        #     df = df.group_by("case_id").agg(Aggregator.get_exprs(df)) 
        return df

    def read_files(regex_path, col_set=None):
        """
        Read multiple parquet files and concat results.
        """
        chunks = []

        for path in glob(str(regex_path)):
            print(f'attempting to read {path}')
            df = pl.read_parquet(path, columns=col_set)
            # TODO: credit buero data is too big, must aggregate here in chunks.
            # df = df.pipe(Pipeline.set_table_dtypes)
            # if depth in [1, 2]:
            #     df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
            chunks.append(df)
            print(f'successfully read {path}')
        print('++++++++++ALL FILES READ+++++++++++++')
        df = pl.concat(chunks, how="vertical_relaxed")
        df = df.unique(subset=["case_id"])
        return df
