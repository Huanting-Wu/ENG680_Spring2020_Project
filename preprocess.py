"""
This is project code for pre-processing.
"""

import pandas as pd

import utils as utils

dow_path = 'data/DJIA_price.csv'
news_path = 'data/headline_news.csv'

df_dow = utils.read_csv_regular(csv_file_path=dow_path)
df_news = utils.read_csv_regular(csv_file_path=news_path)
df_news = df_news.drop(utils.UNWANTED_NEWS_COL_NAME, axis=1)

df_news = utils.clean_byte_encode_in_df(df=df_news, use_col_name_list=utils.TOP_NEWS_COL_NAME_LIST)

cleaned_news_path = utils.export_df_as_csv_ignore_index(df=df_news, export_csv_path='data/headline_news_cleaned.csv')

df_news_cleaned = utils.read_csv_regular(csv_file_path=cleaned_news_path)

df_sentiment = df_news_cleaned[utils.DF_SENTIMENT_COL_NAME_LIST]

sentiment_csv_path = utils.export_df_as_csv_ignore_index(df=df_sentiment, export_csv_path='data/sentiment_data.csv')

df_all = pd.merge(left=df_dow, right=df_sentiment, left_on='Date', right_on='Date', how='inner')

df_all = df_all.rename(utils.LABEL_COL_NAME_MAPPING, axis=1)

df_all_path = utils.export_df_as_csv_ignore_index(df=df_all, export_csv_path='data/model_data.csv')
