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

LABEL_COL_NAME_MAPPING = {'Label': 'Price_Change'}

NEWS_INPUT_PATH = 'data/raw_data/news.csv'

NEWS_CLEANED_EXPORT_PATH = 'data/temp_output/news_cleaned.csv'

NEWS_SENTIMENT_01_EXPORT_PATH = 'data/temp_output/news_sentiment_partial.csv'

SENTIMENT_LABEL_COL_NAME_SUFFIX = '_sentiment'

POSITIVITY_SCORE_COL_NAME_SUFFIX = '_positivity'

NEGATIVITY_SCORE_COL_NAME_SUFFIX = '_negativity'

POLARITY_SCORE_COL_NAME_SUFFIX = '_polarity'

SUBJECTIVITY_SCORE_COL_NAME_SUFFIX = '_subjectivity'

NEWS_SENTIMENT_02_EXPORT_PATH = 'data/temp_output/news_sentiment_full.csv'

PRICE_INPUT_PATH = 'data/raw_data/DJIA_price.csv'

PRICE_LABELED_EXPORT_PATH = 'data/temp_output/price_labeled.csv'

PRICE_CHANGE_BY_COL_NAME = 'Adj Close'

PRICE_CHANGE_VALUE_COL_NAME = 'change_value'

PRICE_CHANGE_DIRECTION_COL_NAME = 'change_direction'

MERGED_DATA_EXPORT_PATH = 'data/price_headline_sentiment.csv'

TIME_STAMP_COL_NAME = 'Date'


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
    df_news = read_csv_regular(csv_file_path=NEWS_INPUT_PATH)
    df_news = clean_byte_encode_in_df(df=df_news, use_col_name_list=TOP_NEWS_COL_NAME_LIST)
    df_news_path = export_df_as_csv_ignore_index(df=df_news, export_csv_path=NEWS_CLEANED_EXPORT_PATH)
    return df_news, df_news_path


def create_sentiment_positivity_negativity(df_news):
    from textblob.en.sentiments import NaiveBayesAnalyzer
    naive_bayes_analyzer = NaiveBayesAnalyzer()

    for news_col_name in TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + '_label_pos_neg'] = [
            naive_bayes_analyzer.analyze(news) for news in
            df_news[news_col_name].astype('str')
        ]

    for news_col_name in TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + SENTIMENT_LABEL_COL_NAME_SUFFIX] = [
            sentiment.classification for sentiment in
            df_news[news_col_name + '_label_pos_neg']
        ]
        df_news[news_col_name + POSITIVITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.p_pos for sentiment in
            df_news[news_col_name + '_label_pos_neg']
        ]
        df_news[news_col_name + NEGATIVITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.p_neg for sentiment in
            df_news[news_col_name + '_label_pos_neg']
        ]

    for col in df_news.columns:
        if '_label_pos_neg' in col:
            df_news = df_news.drop(col, axis=1)

    df_news_path = export_df_as_csv_ignore_index(df=df_news, export_csv_path=NEWS_SENTIMENT_01_EXPORT_PATH)
    return df_news, df_news_path


def create_polarity_subjectivity(df_news):
    from textblob.en.sentiments import PatternAnalyzer
    pattern_analyzer = PatternAnalyzer()

    for news_col_name in TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + '_polar_subjective'] = [
            pattern_analyzer.analyze(news) for news in
            df_news[news_col_name].astype('str')
        ]

    for news_col_name in TOP_NEWS_COL_NAME_LIST:
        df_news[news_col_name + POLARITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.polarity for sentiment in
            df_news[news_col_name + '_polar_subjective']
        ]
        df_news[news_col_name + SUBJECTIVITY_SCORE_COL_NAME_SUFFIX] = [
            sentiment.subjectivity for sentiment in
            df_news[news_col_name + '_polar_subjective']
        ]

    for col in df_news.columns:
        if '_polar_subjective' in col:
            df_news = df_news.drop(col, axis=1)

    df_news_path = export_df_as_csv_ignore_index(df=df_news, export_csv_path=NEWS_SENTIMENT_02_EXPORT_PATH)
    return df_news, df_news_path


def create_price_change_cols():
    df_price = read_csv_regular(csv_file_path=PRICE_INPUT_PATH)
    df_price = df_price.sort_values(TIME_STAMP_COL_NAME, ascending=True)
    df_price[PRICE_CHANGE_VALUE_COL_NAME] = df_price[PRICE_CHANGE_BY_COL_NAME].astype('float').diff()
    df_price[PRICE_CHANGE_DIRECTION_COL_NAME] = np.where(df_price[PRICE_CHANGE_VALUE_COL_NAME] > 0, 1, -1)
    df_price_path = export_df_as_csv_ignore_index(df=df_price, export_csv_path=PRICE_LABELED_EXPORT_PATH)
    return df_price, df_price_path


def merge_format_price_news(df_price, df_news):
    df_price[TIME_STAMP_COL_NAME] = pd.to_datetime(df_price[TIME_STAMP_COL_NAME])
    df_news[TIME_STAMP_COL_NAME] = pd.to_datetime(df_news[TIME_STAMP_COL_NAME])
    df_merged = pd.merge(left=df_price, right=df_news,
                         left_on=TIME_STAMP_COL_NAME, right_on=TIME_STAMP_COL_NAME, how='inner')
    df_merged = df_merged.drop(TOP_NEWS_COL_NAME_LIST, axis=1)

    col_name_order_list = df_merged.columns.to_list()
    col_name_order_list.remove(PRICE_CHANGE_DIRECTION_COL_NAME)
    col_name_order_list.append(PRICE_CHANGE_DIRECTION_COL_NAME)
    df_merged = df_merged[col_name_order_list]

    df_merged_path = export_df_as_csv_ignore_index(df=df_merged, export_csv_path=MERGED_DATA_EXPORT_PATH)
    return df_merged, df_merged_path


numeric_col_name_list = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'Adj Close',
    'change_value',
    'Top1_positivity',
    'Top1_negativity',
    'Top2_positivity',
    'Top2_negativity',
    'Top3_positivity',
    'Top3_negativity',
    'Top4_positivity',
    'Top4_negativity',
    'Top5_positivity',
    'Top5_negativity',
    'Top6_positivity',
    'Top6_negativity',
    'Top7_positivity',
    'Top7_negativity',
    'Top8_positivity',
    'Top8_negativity',
    'Top9_positivity',
    'Top9_negativity',
    'Top10_positivity',
    'Top10_negativity',
    'Top11_positivity',
    'Top11_negativity',
    'Top12_positivity',
    'Top12_negativity',
    'Top13_positivity',
    'Top13_negativity',
    'Top14_positivity',
    'Top14_negativity',
    'Top15_positivity',
    'Top15_negativity',
    'Top16_positivity',
    'Top16_negativity',
    'Top17_positivity',
    'Top17_negativity',
    'Top18_positivity',
    'Top18_negativity',
    'Top19_positivity',
    'Top19_negativity',
    'Top20_positivity',
    'Top20_negativity',
    'Top21_positivity',
    'Top21_negativity',
    'Top22_positivity',
    'Top22_negativity',
    'Top23_positivity',
    'Top23_negativity',
    'Top24_positivity',
    'Top24_negativity',
    'Top25_positivity',
    'Top25_negativity',
    'Top1_polarity',
    'Top1_subjectivity',
    'Top2_polarity',
    'Top2_subjectivity',
    'Top3_polarity',
    'Top3_subjectivity',
    'Top4_polarity',
    'Top4_subjectivity',
    'Top5_polarity',
    'Top5_subjectivity',
    'Top6_polarity',
    'Top6_subjectivity',
    'Top7_polarity',
    'Top7_subjectivity',
    'Top8_polarity',
    'Top8_subjectivity',
    'Top9_polarity',
    'Top9_subjectivity',
    'Top10_polarity',
    'Top10_subjectivity',
    'Top11_polarity',
    'Top11_subjectivity',
    'Top12_polarity',
    'Top12_subjectivity',
    'Top13_polarity',
    'Top13_subjectivity',
    'Top14_polarity',
    'Top14_subjectivity',
    'Top15_polarity',
    'Top15_subjectivity',
    'Top16_polarity',
    'Top16_subjectivity',
    'Top17_polarity',
    'Top17_subjectivity',
    'Top18_polarity',
    'Top18_subjectivity',
    'Top19_polarity',
    'Top19_subjectivity',
    'Top20_polarity',
    'Top20_subjectivity',
    'Top21_polarity',
    'Top21_subjectivity',
    'Top22_polarity',
    'Top22_subjectivity',
    'Top23_polarity',
    'Top23_subjectivity',
    'Top24_polarity',
    'Top24_subjectivity',
    'Top25_polarity',
    'Top25_subjectivity',
    'change_direction'
]

categorical_col_name_list = [
    'Top1_sentiment',
    'Top2_sentiment',
    'Top3_sentiment',
    'Top4_sentiment',
    'Top5_sentiment',
    'Top6_sentiment',
    'Top7_sentiment',
    'Top8_sentiment',
    'Top9_sentiment',
    'Top10_sentiment',
    'Top11_sentiment',
    'Top12_sentiment',
    'Top13_sentiment',
    'Top14_sentiment',
    'Top15_sentiment',
    'Top16_sentiment',
    'Top17_sentiment',
    'Top18_sentiment',
    'Top19_sentiment',
    'Top20_sentiment',
    'Top21_sentiment',
    'Top22_sentiment',
    'Top23_sentiment',
    'Top24_sentiment',
    'Top25_sentiment'
]


def convert_data_type(df_merged):
    df_merged[TIME_STAMP_COL_NAME] = pd.to_datetime(df_merged[TIME_STAMP_COL_NAME])
    df_merged[numeric_col_name_list] = df_merged[numeric_col_name_list].astype('float')
    df_merged[categorical_col_name_list] = df_merged[categorical_col_name_list].astype('str')
    return df_merged
