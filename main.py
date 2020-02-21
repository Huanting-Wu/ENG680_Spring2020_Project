"""
This is main.
"""

from utils import ProjectUtils

if __name__ == '__main__':
    utils = ProjectUtils()

    df_news_cleaned, df_news_cleaned_path = utils.clean_export_news_file()
    print('df_news_cleaned_path:', df_news_cleaned_path)

    df_news_partial, df_news_partial_path = utils.create_sentiment_positivity_negativity(df_news_cleaned)

    print('df_news_partial_path:', df_news_partial_path)

    df_news_full, df_news_full_path = utils.create_polarity_subjectivity(df_news_partial)

    print('df_news_full_path:', df_news_full_path)

    df_price_labeled, df_price_labeled_path = utils.create_price_change_cols()

    print('df_price_labeled_path:', df_price_labeled_path)

    df_merged, df_merged_path = utils.merge_format_price_news(df_price=df_price_labeled, df_news=df_news_full)

    print('df_merged_path:', df_merged_path)

    df_merged = utils.convert_data_type(df_merged=df_merged)