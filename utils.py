"""
This is project utils file.
"""

import numpy as np
import pandas as pd

import reference


class ProjectUtils(object):
    """
    This is project utils class
    """

    def __init__(self):
        self.attribute_list = [func for func in dir(ProjectUtils) if
                               callable(getattr(ProjectUtils, func)) and func.startswith('__')]
        self.method_list = [func for func in dir(ProjectUtils) if
                            callable(getattr(ProjectUtils, func)) and not func.startswith('__')]

    def __clean_byte_encode_in_df(self, df, use_col_name_list):
        """
        Remove byte encoding symbol before strings in data set.
        Parameters
        ----------
        df : pd.DataFrame
            Input Pandas DataFrame.
        use_col_name_list : list of str
            Column names with strings to clean.
        Returns
        -------
        df : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with byte encoding symbol removed.
        """
        for col_name in use_col_name_list:
            df[col_name] = df[col_name].str.replace('b', '')
        return df

    def __read_csv_regular(self, csv_file_path, use_col_list=None):
        """
        Create pd.DataFrame from csv file path.
        Parameters
        ----------
        csv_file_path : str
            Absolute path of input csv file.
        use_col_list : list of str
            Column names with strings to clean.
        Returns
        -------
        df : pd.DataFrame
            pd.DataFrame.
        """
        try:
            df = pd.read_csv(csv_file_path, dtype='str', usecols=use_col_list, sep=',', encoding='utf-8',
                             na_values=None,
                             keep_default_na=True)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file_path, dtype='str', usecols=use_col_list, sep=',', encoding='ISO-8859-1',
                             na_values=None, keep_default_na=True)
        return df

    def __export_df_as_csv_ignore_index(self, df, export_csv_path):
        """
        Export a pd.DataFrame as csv without index column.
        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame. Pandas DataFrame to export.
        export_csv_path : str
            str. Absolute path of export csv file.
        Returns
        -------
        export_csv_path : str
            str. Absolute path of exported csv file.
        """
        df.to_csv(export_csv_path, index=False)
        return export_csv_path

    def clean_export_news_file(self, ):
        """
        Clean and export news data set.
        This is main method for pre-processing news data set.
        Parameters
        ----------

        Returns
        -------
        df_news : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with news data.
        df_news_path : str
            str. Absolute path of export file of news data.
        """
        df_news = self.__read_csv_regular(csv_file_path=reference.NEWS_INPUT_PATH)
        df_news = self.__clean_byte_encode_in_df(df=df_news, use_col_name_list=reference.TOP_NEWS_COL_NAME_LIST)
        df_news_path = self.__export_df_as_csv_ignore_index(df=df_news,
                                                            export_csv_path=reference.NEWS_CLEANED_EXPORT_PATH)
        return df_news, df_news_path

    def create_sentiment_positivity_negativity(self, df_news):
        """
        Create sentiment label, positivity scores, negativity scores in news data set.
        Parameters
        ----------
        df_news : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with news data.
        Returns
        -------
        df_news : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with news data.
        df_news_path : str
            str. Absolute path of export file of news data.
        """
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

        df_news_path = self.__export_df_as_csv_ignore_index(df=df_news,
                                                            export_csv_path=reference.NEWS_SENTIMENT_01_EXPORT_PATH)
        return df_news, df_news_path

    def create_polarity_subjectivity(self, df_news):
        """
        Create polarity scores and subjectivity scores in news data set.
        Parameters
        ----------
        df_news : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with news data.
        Returns
        -------
        df_news : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with news data.
        df_news_path : str
            str. Absolute path of export file of news data.
        """
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

        df_news_path = self.__export_df_as_csv_ignore_index(df=df_news,
                                                            export_csv_path=reference.NEWS_SENTIMENT_02_EXPORT_PATH)
        return df_news, df_news_path

    def create_price_change_cols(self, ):
        """
        Create price change column in price data set.
        Returns
        -------
        df_price : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with price data.
        df_price_path : str
            str. Absolute path of export file of price data.
        """
        df_price = self.__read_csv_regular(csv_file_path=reference.PRICE_INPUT_PATH)
        df_price = df_price.sort_values(reference.TIME_STAMP_COL_NAME, ascending=True)
        df_price[reference.PRICE_CHANGE_VALUE_COL_NAME] = df_price[reference.PRICE_CHANGE_BY_COL_NAME].astype(
            'float').diff()
        df_price[reference.PRICE_CHANGE_DIRECTION_COL_NAME] = np.where(
            df_price[reference.PRICE_CHANGE_VALUE_COL_NAME] > 0,
            1, 0)
        df_price_path = self.__export_df_as_csv_ignore_index(df=df_price,
                                                             export_csv_path=reference.PRICE_LABELED_EXPORT_PATH)
        return df_price, df_price_path

    def merge_format_price_news(self, df_price, df_news):
        """
        Merge sentiment data set to price data set.
        Parameters
        ----------
        df_price : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with price data.
        df_news : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with news data.
        Returns
        -------
        df_merged : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with merged data.
        df_merged_path : str
            str. Absolute path of exported file with merged data.
        """
        df_price[reference.TIME_STAMP_COL_NAME] = pd.to_datetime(df_price[reference.TIME_STAMP_COL_NAME])
        df_news[reference.TIME_STAMP_COL_NAME] = pd.to_datetime(df_news[reference.TIME_STAMP_COL_NAME])
        df_merged = pd.merge(left=df_price, right=df_news,
                             left_on=reference.TIME_STAMP_COL_NAME, right_on=reference.TIME_STAMP_COL_NAME, how='inner')
        df_merged = df_merged.drop(reference.TOP_NEWS_COL_NAME_LIST, axis=1)

        col_name_order_list = df_merged.columns.to_list()
        col_name_order_list.remove(reference.PRICE_CHANGE_DIRECTION_COL_NAME)
        col_name_order_list.append(reference.PRICE_CHANGE_DIRECTION_COL_NAME)
        df_merged = df_merged[col_name_order_list]

        df_merged_path = self.__export_df_as_csv_ignore_index(df=df_merged,
                                                              export_csv_path=reference.MERGED_DATA_EXPORT_PATH)
        return df_merged, df_merged_path

    def convert_data_type(self, df_merged):
        """
        Convert data type in merged data set for modeling.
        Parameters
        ----------
        df_merged : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with merged data.
        Returns
        -------
        df_merged : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with converted data types.
        """
        df_merged[reference.TIME_STAMP_COL_NAME] = pd.to_datetime(df_merged[reference.TIME_STAMP_COL_NAME])
        df_merged[reference.NUMERIC_COL_NAME_LIST] = df_merged[reference.NUMERIC_COL_NAME_LIST].astype('float')
        df_merged[reference.CATEGORICAL_COL_NAME_LIST] = df_merged[reference.CATEGORICAL_COL_NAME_LIST].astype('str')
        return df_merged

    def normalize_df_merged(self, df_merged):
        """
        Normalize feature columns.
        Continuous numerical columns will conform to a standard Gaussian distribution.
        Categorical columns will conform to a binary encoding.
        Parameters
        ----------
        df_merged : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with merged data.
        Returns
        -------
        df_normalized : pd.DataFrame
            pd.DataFrame. Pandas DataFrame with normalized data.
        df_normalized_path : str
            str. Absolute path of exported file with normalized data.
        """
        df_normalized = df_merged.copy()
        from sklearn.preprocessing import StandardScaler
        standard_scaler = StandardScaler()
        df_normalized[reference.NUMERIC_COL_NAME_LIST] = (
            standard_scaler.fit_transform(df_normalized[reference.NUMERIC_COL_NAME_LIST]))
        for categorical_col_name in reference.CATEGORICAL_COL_NAME_LIST:
            df_normalized[categorical_col_name] = np.where(df_normalized[categorical_col_name] == 'pos', 1, 0)
        df_normalized_path = self.__export_df_as_csv_ignore_index(
            df=df_normalized,
            export_csv_path=reference.NORMALIZED_DATA_EXPORT_PATH)
        return df_normalized, df_normalized_path

    def read_split_model_data(self, model_data_path):
        """
        Read model data from path and split data into x and y sets, and train, dev, and test sets.
        Parameters
        ----------
        model_data_path : str
            str. Absolute file path of model data.
        Returns
        -------
        x_train : pd.DataFrame
            pd.DataFrame. X training set.
        x_dev : pd.DataFrame
            pd.DataFrame. X development set.
        x_test : pd.DataFrame
            pd.DataFrame. X testing set.
        y_train : pd.DataFrame
            pd.DataFrame. Y training set.
        y_dev : pd.DataFrame
            pd.DataFrame. Y development set.
        y_test : pd.DataFrame
            pd.DataFrame. Y testing set.
        """
        df = pd.read_csv(model_data_path).fillna(0)

        X = df[df.columns[1:-1]]
        y = df[df.columns[-1]]
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        x_train, x_dev, x_test = np.split(X, [int(.8 * len(X)), int(.9 * len(X))])
        print("X_train shape:", x_train.shape)
        print("X_dev shape:", x_dev.shape)
        print("X_test shape:", x_test.shape)

        y_train, y_dev, y_test = np.split(y, [int(.8 * len(y)), int(.9 * len(y))])
        print("y_train shape:", y_train.shape)
        print("y_dev shape:", y_dev.shape)
        print("y_test shape:", y_test.shape)
        return x_train, x_dev, x_test, y_train, y_dev, y_test

    def get_classifier_dict(self):
        """
        Create a dictionary of key-value pair as `model_name : model object`.
        Returns
        -------
        model_dict : dict
            dict. A dictionary of key-value pair as `model_name : model object`.
        """
        from sklearn.linear_model import LogisticRegression
        logistic_regression = LogisticRegression(random_state=42, solver="lbfgs")
        from sklearn.ensemble import RandomForestClassifier
        random_forest = RandomForestClassifier(random_state=42)
        from sklearn.ensemble import GradientBoostingClassifier
        gradient_boosting = GradientBoostingClassifier(random_state=42)
        model_dict = {"logistic regression": logistic_regression,
                      "random forest": random_forest,
                      "gradient boosting": gradient_boosting
                      }
        return model_dict

    def __measure_accuracy(self, trained_model, x_dev, x_test, y_dev, y_test):
        """
        Measure accuracy score as probability from a trained model on dev and test sets.
        Parameters
        ----------
        trained_model : sklearn object
            sklearn object. A model that is fitted on training set.
        x_dev : pd.DataFrame
            pd.DataFrame. X development set.
        x_test : pd.DataFrame
            pd.DataFrame. X testing set.
        y_dev : pd.DataFrame
            pd.DataFrame. Y development set.
        y_test : pd.DataFrame
            pd.DataFrame. Y testing set.
        Returns
        -------
        accuracy_dev : float
            float. Accuracy score on development set as probability
        accuracy_test : float
            float. Accuracy score on testing set as probability
        """
        accuracy_dev = trained_model.score(x_dev, y_dev)
        accuracy_test = trained_model.score(x_test, y_test)
        return accuracy_dev, accuracy_test

    def __get_confusion_matrix(self, trained_model, x_dev, x_test, y_dev, y_test):
        """
        Get confusion matrix of a trained model for dev and test set.
        Parameters
        ----------
        trained_model : sklearn object
            sklearn object. A model that is fitted on training set.
        x_dev : pd.DataFrame
            pd.DataFrame. X development set.
        x_test : pd.DataFrame
            pd.DataFrame. X testing set.
        y_dev : pd.DataFrame
            pd.DataFrame. Y development set.
        y_test : pd.DataFrame
            pd.DataFrame. Y testing set.
        Returns
        -------
        confusion_matrix_dev : np.array
            np.array. Confusion matrix on dev set.
        confusion_matrix_test : np.array
            np.array. Confusion matrix on test set.
        """
        from sklearn.metrics import confusion_matrix
        y_dev_pred = trained_model.predict(x_dev)
        y_test_pred = trained_model.predict(x_test)
        confusion_matrix_dev = confusion_matrix(y_dev, y_dev_pred)
        confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
        return confusion_matrix_dev, confusion_matrix_test

    def fit_and_measure(self, x_train, x_dev, x_test, y_train, y_dev, y_test):
        """
        Fit a model and measure its accuracy and confusion matrix.
        This is main method for prediction.
        Parameters
        ----------
        x_train : pd.DataFrame
            pd.DataFrame. X training set.
        x_dev : pd.DataFrame
            pd.DataFrame. X development set.
        x_test : pd.DataFrame
            pd.DataFrame. X testing set.
        y_train : pd.DataFrame
            pd.DataFrame. Y training set.
        y_dev : pd.DataFrame
            pd.DataFrame. Y development set.
        y_test : pd.DataFrame
            pd.DataFrame. Y testing set.
        Returns
        -------
        accuracy_dict : dict
            dict. Dictionary of key-value pair in format
            `"model name": {"accuracy dev": accuracy_dev, "accuracy test": accuracy_test}
        confusion_matrix_dict : dict
            dict. Dictionary of key-value pair in format
            `"model name": {"confusion matrix dev": accuracy_dev, "confusion matrix test": accuracy_test}
        """
        model_dict = self.get_classifier_dict()
        accuracy_dict = {}
        confusion_matrix_dict = {}
        for model_name, model in zip(model_dict.keys(), model_dict.values()):
            print("model name:", model_name)
            trained_model = model.fit(x_train, y_train)
            accuracy_dev, accuracy_test = self.__measure_accuracy(
                trained_model=trained_model,
                x_dev=x_dev,
                x_test=x_test,
                y_dev=y_dev,
                y_test=y_test
            )
            print("accuracy dev set:", accuracy_dev)
            print("accuracy test set:", accuracy_test)
            accuracy_dict.update({model_name: {"accuracy dev set": accuracy_dev,
                                               "accuracy test set": accuracy_test}})
            confusion_matrix_dev, confusion_matrix_test = self.__get_confusion_matrix(
                trained_model=trained_model,
                x_dev=x_dev,
                x_test=x_test,
                y_dev=y_dev,
                y_test=y_test
            )
            print("consusion matrix dev set:\n", confusion_matrix_dev)
            print("consusion matrix test set:\n", confusion_matrix_test)
            confusion_matrix_dict.update({model_name: {"confusion matrix dev set": confusion_matrix_dev,
                                                       "confusion matrix test set": confusion_matrix_test}})
        return accuracy_dict, confusion_matrix_dict
