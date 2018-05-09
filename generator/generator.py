"""
Fake data generator.
"""

import datetime
import os
from typing import Dict

import collections
import numpy as np
import pandas as pd


# Generic type definitions.
ndist_params = collections.namedtuple('ndist_params', ('mu', 'sigma', 'derives_from', 'decimals'))

#
# Generator settings
#

# Base paths.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Input files.
AD_GROUP_NAMES_FILE = os.path.join(DATA_DIR, 'gen_ad_groups.csv')
WEEKLY_PERF_FILE = os.path.join(DATA_DIR, 'gen_weekly_perf.csv')
WEEKDAY_PERF_FILE = os.path.join(DATA_DIR, 'gen_weekday_perf.csv')

# Settings for the random generator.
METRICS_RAND_SETTINGS: Dict[str, ndist_params] = {
    'Impressions': ndist_params(mu=200, sigma=40, derives_from=None, decimals=0),
    'Clicks': ndist_params(mu=0.1, sigma=0.01, derives_from='Impressions', decimals=0),
    'Cost': ndist_params(mu=5, sigma=1, derives_from='Clicks', decimals=2),
    'Conversions': ndist_params(mu=0.1, sigma=0.02, derives_from='Clicks', decimals=0),
    'ConversionsValue': ndist_params(mu=1500, sigma=500, derives_from='Conversions', decimals=2),
}
HIGH_QUALITY_SCORE_SETTINGS = ndist_params(mu=7, sigma=2, derives_from=None, decimals=0)
LOW_QUALITY_SCORE_SETTINGS = ndist_params(mu=4, sigma=2, derives_from=None, decimals=0)
KEYWORD_IMPRESSIONS_SETTINGS = ndist_params(mu=500, sigma=300, derives_from=None, decimals=0)

# Simulated days without credit.
DAYS_WITHOUT_CREDIT = {
    datetime.datetime(2018, 3, 17),
    datetime.datetime(2018, 3, 18),
}

# Output files.
AD_GROUP_DATA_FILE = os.path.join(DATA_DIR, 'data_ad_group_performance.xlsx')
QUALITY_SCORE_DATA_FILE = os.path.join(DATA_DIR, 'data_keywords_quality_score.xlsx')


def load_weekday_perf(filename) -> pd.DataFrame:
    """
    Loads the data file with source week days.

    :param filename: File path.
    :return: Loaded DataFrame.
    """
    return pd.read_csv(filename, header=0)


def load_weekly_perf(filename) -> pd.DataFrame:
    """
    Loads the data file with source weekly performance.

    :param filename: File path
    :return: Loaded DataFrame.
    """
    weekly_perf = pd.read_csv(filename, header=0)
    weekly_perf['iso_week'] = pd.to_datetime(weekly_perf['iso_week'], format='%YW%W-%w')
    return weekly_perf


def load_ad_groups(filename) -> pd.DataFrame:
    """
    Loads the data file with ad groups.

    :param filename: File path.
    :return: Loaded DataFrame.
    """
    return pd.read_csv(filename, header=0)


def generate_ad_group_performance(ad_groups: pd.DataFrame, weekly_perf: pd.DataFrame, weekday_perf: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Generates a data set with ad group daily performance.

    :param ad_groups: Ad groups.
    :param weekly_perf: Performance for each week.
    :param weekday_perf: Performance for each week day.
    :return: Generated DataFrame.
    """

    # Join the tables.
    result: pd.DataFrame = pd.merge(ad_groups, weekly_perf, on='key', how='inner')
    result: pd.DataFrame = pd.merge(result, weekday_perf, on='key', how='inner')
    result.drop(columns=['key'], inplace=True)

    # Convert week date and day offset to concrete days.
    result['days_delta'] = pd.to_timedelta(result['iso_weekday'], unit='D')
    result['Date'] = result['iso_week'] + result['days_delta']

    days_without_credit_filter = result['Date'].isin(DAYS_WITHOUT_CREDIT)

    for column_name, rand_params in METRICS_RAND_SETTINGS.items():
        random_seq = generate_metric_column(column_name, rand_params, result)
        random_seq[days_without_credit_filter] = 0

        result[column_name] = random_seq

    fields = ['CampaignId', 'CampaignName', 'AdGroupId', 'AdGroupName', 'Date'] + \
             [metric for metric in METRICS_RAND_SETTINGS.keys()]
    return result[fields]


def generate_metric_column(column_name, rand_params, result):
    """
    Generates a series with values for the specified metrics.

    If the result set contains any of the following columns, they will be used as coefficients for the random value:

    * f'weekly_perf_{column_name}'
    * f'weekday_perf_{column_name}'
    * f'ad_group_perf_{column_name}'

    :param column_name: Name of the column.
    :param rand_params: Parameters of the random distribution.
    :param result: Result set.
    :return: Generated series.
    """

    random_seq = np.random.normal(rand_params.mu, rand_params.sigma, size=len(result))
    random_seq[random_seq < 0] = 0

    if rand_params.derives_from:
        random_seq *= result[rand_params.derives_from]
    if f'weekly_perf_{column_name}' in result:
        random_seq *= result[f'weekly_perf_{column_name}']
    if f'weekday_perf_{column_name}' in result:
        random_seq *= result[f'weekday_perf_{column_name}']
    if f'ad_group_perf_{column_name}' in result:
        random_seq *= result[f'ad_group_perf_{column_name}']
    if rand_params.decimals == 0:
        random_seq = random_seq.astype(np.int64)
    else:
        random_seq = random_seq.round(rand_params.decimals)
    return random_seq


def generate_quality_score(rand_params: ndist_params, size):
    """
    Generates a sequence of Quality Score values using normal distribution of specified parameters.

    :param rand_params: Parameters of the random distribution.
    :param size: Number of rows.
    :return: Generated series.
    """

    random_seq = np.random.normal(rand_params.mu, rand_params.sigma, size=size)
    random_seq[random_seq > 10] = 10
    random_seq[random_seq < 1] = 1
    return random_seq.astype(np.int64)


def generate_keywords_quality_score(ad_groups: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a list of keywords with Quality Score for each Ad Group.

    :param ad_groups: List of all Ad Groups.
    :return: Generated DataFrame.
    """
    keywords = pd.DataFrame({
        'key': 1,
        'kw_base': 'Keyword #',
        'kw_num': np.arange(1, 200),
    })
    keywords['Keyword'] = keywords['kw_base'] + keywords['kw_num'].astype(str)
    keywords.drop(columns=['kw_base', 'kw_num'], inplace=True)

    result: pd.DataFrame = pd.merge(ad_groups, keywords, on='key', how='inner')
    result.drop(columns=['key'], inplace=True)

    high_qs = generate_quality_score(HIGH_QUALITY_SCORE_SETTINGS, len(result))
    low_qs = generate_quality_score(LOW_QUALITY_SCORE_SETTINGS, len(result))

    selection_cond = ((result['AdGroupId'] % 2) == 0)
    result['QualityScore'] = np.where(selection_cond, high_qs, low_qs)
    result['Impressions'] = generate_metric_column('Impressions', KEYWORD_IMPRESSIONS_SETTINGS, result)

    fields = ['CampaignId', 'CampaignName', 'AdGroupId', 'AdGroupName', 'Keyword', 'Impressions', 'QualityScore']

    return result[fields]


def main():
    """
    Generates all required data sets.
    """
    # Load source data for generator.
    ad_groups = load_ad_groups(AD_GROUP_NAMES_FILE)
    weekly_perf = load_weekly_perf(WEEKLY_PERF_FILE)
    weekday_perf = load_weekday_perf(WEEKDAY_PERF_FILE)

    # Add joining key which will be used to produce the cartesian product:
    ad_groups['key'] = 1
    weekly_perf['key'] = 1
    weekday_perf['key'] = 1

    # Generate Ad Group performance.
    ad_group_performance = generate_ad_group_performance(ad_groups, weekly_perf, weekday_perf)
    ad_group_performance.to_excel(AD_GROUP_DATA_FILE, sheet_name='data', index=False)

    # Generate Keywords with Quality Score.
    quality_score = generate_keywords_quality_score(ad_groups)
    quality_score.to_excel(QUALITY_SCORE_DATA_FILE, sheet_name='quality_score', index=False)


if __name__ == '__main__':
    main()
