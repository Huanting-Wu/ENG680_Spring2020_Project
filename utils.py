"""
This is project utils.
"""

import pandas as pd
import numpy as np

import reference


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


def export_df_as_csv_ignore_index(df, export_csv_path):
    df.to_csv(export_csv_path, index=False)
    return export_csv_path


def clean_export_news_file():
    df_news = read_csv_regular(csv_file_path=reference.NEWS_INPUT_PATH)
    df_news = clean_byte_encode_in_df(df=df_news, use_col_name_list=reference.TOP_NEWS_COL_NAME_LIST)
    df_news_path = export_df_as_csv_ignore_index(df=df_news, export_csv_path=reference.NEWS_CLEANED_EXPORT_PATH)
    return df_news, df_news_path


def create_sentiment_positivity_negativity(df_news):
    from textblob.en.sentiments import NaiveBayesAnalyzer
    naive_bayes_analyzer = NaiveBayesAnalyzer()

    for news_col_name in reference.TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + '_label_pos_neg'] = [
            naive_bayes_analyzer.analyze(news) for news in
            df_news[news_col_name].astype('str')
        ]

    for news_col_name in reference.TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + reference.SENTIMENT_LABEL_COL_NAME_SUFFIX] = [
            sentiment.classification for sentiment in
            df_news[news_col_name + '_label_pos_neg']
        ]
        df_news[news_col_name + reference.POSITIVITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.p_pos for sentiment in
            df_news[news_col_name + '_label_pos_neg']
        ]
        df_news[news_col_name + reference.NEGATIVITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.p_neg for sentiment in
            df_news[news_col_name + '_label_pos_neg']
        ]

    for col in df_news.columns:
        if '_label_pos_neg' in col:
            df_news = df_news.drop(col, axis=1)

    df_news_path = export_df_as_csv_ignore_index(df=df_news, export_csv_path=reference.NEWS_SENTIMENT_01_EXPORT_PATH)
    return df_news, df_news_path


def create_polarity_subjectivity(df_news):
    from textblob.en.sentiments import PatternAnalyzer
    pattern_analyzer = PatternAnalyzer()

    for news_col_name in reference.TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + '_polar_subjective'] = [
            pattern_analyzer.analyze(news) for news in
            df_news[news_col_name].astype('str')
        ]

    for news_col_name in reference.TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + reference.POLARITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.polarity for sentiment in
            df_news[news_col_name + '_polar_subjective']
        ]
        df_news[news_col_name + reference.SUBJECTIVITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.subjectivity for sentiment in
            df_news[news_col_name + '_polar_subjective']
        ]

    for col in df_news.columns:
        if '_polar_subjective' in col:
            df_news = df_news.drop(col, axis=1)

    df_news_path = export_df_as_csv_ignore_index(df=df_news, export_csv_path=reference.NEWS_SENTIMENT_02_EXPORT_PATH)
    return df_news, df_news_path


def create_price_change_cols():
    df_price = read_csv_regular(csv_file_path=reference.PRICE_INPUT_PATH)
    df_price = df_price.sort_values(reference.TIME_STAMP_COL_NAME, ascending=True)
    df_price[reference.PRICE_CHANGE_VALUE_COL_NAME] = df_price[reference.PRICE_CHANGE_BY_COL_NAME].astype(
        'float').diff()
    df_price[reference.PRICE_CHANGE_DIRECTION_COL_NAME] = np.where(df_price[reference.PRICE_CHANGE_VALUE_COL_NAME] > 0,
                                                                   1, -1)
    df_price_path = export_df_as_csv_ignore_index(df=df_price, export_csv_path=reference.PRICE_LABELED_EXPORT_PATH)
    return df_price, df_price_path


def merge_format_price_news(df_price, df_news):
    df_price[reference.TIME_STAMP_COL_NAME] = pd.to_datetime(df_price[reference.TIME_STAMP_COL_NAME])
    df_news[reference.TIME_STAMP_COL_NAME] = pd.to_datetime(df_news[reference.TIME_STAMP_COL_NAME])
    df_merged = pd.merge(left=df_price, right=df_news,
                         left_on=reference.TIME_STAMP_COL_NAME, right_on=reference.TIME_STAMP_COL_NAME, how='inner')
    df_merged = df_merged.drop(reference.TOP_NEWS_COL_NAME_LIST, axis=1)

    col_name_order_list = df_merged.columns.to_list()
    col_name_order_list.remove(reference.PRICE_CHANGE_DIRECTION_COL_NAME)
    col_name_order_list.append(reference.PRICE_CHANGE_DIRECTION_COL_NAME)
    df_merged = df_merged[col_name_order_list]

    df_merged_path = export_df_as_csv_ignore_index(df=df_merged, export_csv_path=reference.MERGED_DATA_EXPORT_PATH)
    return df_merged, df_merged_path


def convert_data_type(df_merged):
    df_merged[reference.TIME_STAMP_COL_NAME] = pd.to_datetime(df_merged[reference.TIME_STAMP_COL_NAME])
    df_merged[reference.NUMERIC_COL_NAME_LIST] = df_merged[reference.NUMERIC_COL_NAME_LIST].astype('float')
    df_merged[reference.CATEGORICAL_COL_NAME_LIST] = df_merged[reference.CATEGORICAL_COL_NAME_LIST].astype('str')
    return df_merged
