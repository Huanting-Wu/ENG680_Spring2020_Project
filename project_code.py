"""
This is project code.
"""

import project_utils as utils

dow_path = 'data/DJIA_price.csv'
news_path = 'data/headline_news.csv'

df_dow = utils.read_csv_regular(csv_file_path=dow_path)
df_news = utils.read_csv_regular(csv_file_path=news_path)
df_news = df_news.drop('Unnamed: 0', axis=1)

print(df_dow.head())

print(df_news.head())
