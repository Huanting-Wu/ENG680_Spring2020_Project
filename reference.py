"""
This is project reference.
"""

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

NUMERIC_COL_NAME_LIST = [
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

CATEGORICAL_COL_NAME_LIST = [
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
