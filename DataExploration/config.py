from enum import Enum

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