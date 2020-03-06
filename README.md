# SFSU Applied Computational Linguistics 
# Spring 2020 Class Project

## Team Member:
Tyler Anderson 

Sheetal Kalburgi

Hoa Nguyen

Jasjit Samra

Huanting Wu

## Applying NLP on News Headlines for Stock Market Prediction

The volatility of the stock market is hard to predict, but it is not impossible. By using a Natural Language Processing 
(NLP) based model that uses news article headlines to predict the movements in the stock market, we will be able to 
predict volatility. Our research aims to understand how to apply Computational Linguistics to perform sentiment analysis 
on news headlines to predict stock market movement in Dow Jones Industrial Average (DJIA).  

## Literature Review
One of the earliest promising results comes from Mining and Summarizing Customer Reviews, this study uses a dataset of 
customer reviews to generate probabilities of positive and negative sentiment labels by identifying adjective words, 
determining each words semantic orientation, and deciding an opinion orientation for each sentence (Hu & Liu, 2004). 
The algorithm for the binary pair of sentiment labels with positivity and negativity is further augmented by the 
generation of <pros> and <cons> tags for positive and negative comparative sentiments. (Ganapathibhotla & Liu, 2008).

Then, in the 2010s, algorithms that can generate polarity and subjectivity sentiments started to mature. Using the MPQA 
and Subjectivity Lexicon datasets, a comprehensive algorithm is trained successfully to classify sentiments for a corpus 
of opinion texts (Jain & Nemade, 2010). Generation of opinion related sentiments is further perfected by one of the 
first comprehensive sentiment analysis on a corpus of tweets on social media Twitter (Sanders, 2011). With a more 
encompassing Twitter dataset curated, the algorithm is further trained on this dataset to obtain better classification 
results on news headlines and the public’s reactions to these news headlines (Saif, Fernandez, He & Alani, 2013).

## Data Sources and Technical Approach
Data sets for this project are obtained from the Daily News for Stock Market Prediction dataset uploaded to Kaggle 
(Sun, 2019) which is crawled from Reddit WorldNews Channel. We will use parts of the Dow Jones Industrial Average (DJIA) 
price data set and parts of the Headline News data set. Timeframe of the data set is from 2008-06-08 to 2016-07-01. 

We will attempt to generate sentiments on the 25 headlines for that day. We will use the TextBlob package in Python to 
generate the sentiment features (Sloria). The following features will be created: sentiment label, positivity score, 
negativity score, polarity score, subjectivity score. 

After sentiment features are created, we will drop the 25 headline columns. Then we will perform a left join from the 
news sentiment data set to the Dow price data set on the Date column. This merged data set will be the data source for 
our model. We will then normalize the data set to uniform the scales of values across columns. 

For the modelling, we will first use various classifiers with their default parameters to predict whether Dow's closing 
price will increase or decrease at the end of that trading day. Depending on the accuracy of the default models, we may 
want to further customize our models to achieve better accuracy.

## Project Contribution to the Field
Our model could be used by hedge funds or mutual funds that actively manage portfolios or investment companies to 
understand how their actions and earnings reports are interpreted by the public.The stock market can be impacted by 
various factors, irrespective of the industry, such as a medical emergency like the COVID-19 outbreak or changing 
political environment in the country, making predictions of the stock market difficult. Hence, by applying such 
techniques and models on current news, it will be possible to predict stock market movement more accurately. 

## Reference
Ganapathibhotla, M., & Liu, B. (2008). Mining opinions in comparative sentences. Proceedings of the 22nd International 
    Conference on Computational Linguistics - COLING 08. doi: 10.3115/1599081.1599112
    
Hu, M., & Liu, B. (2004). Mining and summarizing customer reviews. Proceedings of the 2004 ACM SIGKDD International 
    Conference on Knowledge Discovery and Data Mining - KDD 04. doi: 10.1145/1014052.1014073
    
Jain, T. I., & Nemade, D. (2010). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. International 
    Journal of Computer Applications, 7(5), 12–21. doi: 10.5120/1160-1453
    
Saif, H., Fernandez, M., He, Y., & Alani, H. (2013). Evaluation datasets for Twitter sentiment analysis: a survey and 
    a new dataset, the STS-Gold.
    
Sanders, N.J. (2011) Sanders-Twitter Sentiment Corpus. Sanders Analytics LLC.

Sloria. (2020, January 15). sloria/TextBlob. Retrieved February 21, 2020, from https://github.com/sloria/TextBlob

Sun, J. (2019, November 13). Daily News for Stock Market Prediction. Retrieved February 17, 2020, from 
    https://www.kaggle.com/aaron7sun/stocknews 



