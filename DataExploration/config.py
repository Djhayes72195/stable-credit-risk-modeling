from enum import Enum
from pathlib import Path

class FeaturePartitionEnum(Enum):
    APPLPREV = 'applprev'
    CREDIT_BUREAU = 'credit_bureau'
    DEBITCARD = 'debitcard'
    DEPOSIT = 'deposit'
    OTHER = 'other'
    PERSON = 'person'
    STATIC = 'static'
    TAX_REGISTRY = 'tax_registry'

BUCKET_NAME = "credit-risk-modeling-dj72195"
ROOT = Path('/Volumes/My Passport for Mac/CreditRiskModeling') # TODO: Use global_config instead.
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
CORE_COLUMNS = ["target", "case_id", "WEEK_NUM", "num_group1", "num_group2"]

FEATURE_REPORT_PATH = Path('/Users/dustinhayes/Desktop/GitHub/stable-credit-risk-modeling/Data/FeatureReport')


# Paths with * map to multiple files that should be proecessed together
data_store = {
    "df_base": TRAIN_DIR / "train_base.parquet",
    "depth_0": [
        TRAIN_DIR / "train_static_cb_0.parquet",
        TRAIN_DIR / "train_static_0_*.parquet",
    ],
    "depth_1": [
        TRAIN_DIR / "train_applprev_1_*.parquet",
        TRAIN_DIR / "train_tax_registry_a_1.parquet",
        TRAIN_DIR / "train_tax_registry_b_1.parquet",
        TRAIN_DIR / "train_tax_registry_c_1.parquet",
        TRAIN_DIR / "train_credit_bureau_a_1_*.parquet",
        TRAIN_DIR / "train_credit_bureau_b_1.parquet",
        TRAIN_DIR / "train_other_1.parquet",
        TRAIN_DIR / "train_person_1.parquet",
        TRAIN_DIR / "train_deposit_1.parquet",
        TRAIN_DIR / "train_debitcard_1.parquet",
    ],
    "depth_2": [
        TRAIN_DIR / "train_credit_bureau_b_2.parquet",
        TRAIN_DIR / "train_credit_bureau_a_2_*.parquet",
        TRAIN_DIR / "train_applprev_2.parquet",
        TRAIN_DIR / "train_person_2.parquet"
    ]
}
