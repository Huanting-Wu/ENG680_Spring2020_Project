"""
This is project utils.
"""

import pandas as pd
import numpy as np

TOP_NEWS_COL_NAME_LIST = [
    'Top1',
    'Top2',
    'Top3',
    'Top4',
    'Top5',
    'Top6',
    'Top7',
    'Top8',
    'Top9',
    'Top10',
    'Top11',
    'Top12',
    'Top13',
    'Top14',
    'Top15',
    'Top16',
    'Top17',
    'Top18',
    'Top19',
    'Top20',
    'Top21',
    'Top22',
    'Top23',
    'Top24',
    'Top25',
]


def clean_byte_encode_in_df(df, use_col_name_list):
    for col_name in use_col_name_list:
        df[col_name] = df[col_name].str.replace('b', '')
    return df


def read_csv_regular(csv_file_path, use_col_list=None):
    try:
        df = pd.read_csv(csv_file_path, dtype='str', usecols=use_col_list, sep=',', encoding='utf-8', na_values=None,
                         keep_default_na=True)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, dtype='str', usecols=use_col_list, sep=',', encoding='ISO-8859-1',
                         na_values=None, keep_default_na=True)
    return df
