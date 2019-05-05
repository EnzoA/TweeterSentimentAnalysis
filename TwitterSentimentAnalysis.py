import numpy as np
import pandas as pd
import tweepy
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import operator

class TweeterSentimentAnalyzer:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
        self.retrieved_tweets = None
        self.topic = None

    def get_tweets_polarity(self, topic, count, lang='en'):
        self.topic = topic
        tweets = self.api.search(topic, count=count, lang=lang)
        tweets_df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
        sid = SentimentIntensityAnalyzer()
        listy = []
        for index, row in tweets_df.iterrows():
            scores = sid.polarity_scores(row["Tweets"])
            listy.append(scores)
        series = pd.Series(listy)
        tweets_df['polarity'] = series.values
        self.retrieved_tweets = tweets_df

    def plot_tweets_polarities_count(self):
        labels = ['Negative', 'Neutral', 'Positive']
        indexes = np.arange(len(labels))

        negative_tweets_count = 0
        neutral_tweets_count = 0
        positive_tweets_count = 0

        for i in self.retrieved_tweets.index:
            row = self.retrieved_tweets.iloc[i]
            polarity = max(row['polarity'].items(), key=operator.itemgetter(1))[0]
            if polarity is 'neg':
                negative_tweets_count += 1
            elif polarity is 'neu':
                neutral_tweets_count += 1
            elif polarity is 'pos':
                positive_tweets_count += 1
        
        plt.bar(indexes, [negative_tweets_count, neutral_tweets_count, positive_tweets_count])
        plt.xlabel('Polarity', fontsize=10)
        plt.ylabel('Number of tweets', fontsize=10)
        plt.xticks(indexes, labels, fontsize=10, rotation=30)
        plt.title('Tweets about {0} polarities'.format(self.topic))
        plt.show()

    def plot_tweets_compound_polarities(self):
        scores = [polarity['compound'] for polarity in self.retrieved_tweets['polarity']]
        plt.hist(scores, bins=20, color='blue', edgecolor='black', density=True, label='Compound polarities')
        plt.xlabel('Compound polarity')
        plt.ylabel('Tweets')
        plt.title('Compound polarities for {0}'.format(self.topic))
        plt.show()

tweeter_sentiment_analyzer = TweeterSentimentAnalyzer(consumer_key='YOUR CONSUMER KEY',
                                                      consumer_secret='YOUR CONSUMER SECRET',
                                                      access_token='YOUR ACCESS TOKEN',
                                                      access_token_secret='YOUR TOKEN SECRET')

tweeter_sentiment_analyzer.get_tweets_polarity('deep learning', count=1000, lang='en')
tweeter_sentiment_analyzer.plot_tweets_compound_polarities()

    