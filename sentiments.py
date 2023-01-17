from flask import Blueprint, render_template, request
import matplotlib.pyplot as plt
import os
import pickle
import tweepy
import csv
import re
from textblob import TextBlob
import matplotlib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from keras.utils import pad_sequences
import string
import numpy as np
import pandas as pd

matplotlib.use('agg')

# register this file as a blueprint
second = Blueprint("second", __name__, static_folder="static",
                   template_folder="template")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
gru_model = load_model('main_GRU_model.h5')
cnn_model = load_model('main_CNN_model.h5')
lstm_model = load_model('main_LSTM_model.h5')
max_sequence_length = 300

# render page when url is called
@second.route("/sentiment_analyzer")
def sentiment_analyzer():
    return render_template("sentiment_analyzer.html")


# class with main logic
class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []
        self.rows = []
        self.table = []

    # This function first connects to the Tweepy API using API keys
    def DownloadData(self, keyword, tweets):

        # authenticating
        consumerKey = 'Mn0WQwlUr1Sx11ckCTdrSGSww'
        consumerSecret = '6xcxy5FW1M1cGNV2KyRPV0i88xz8ub0QdECZVJIVFtP3BvCYg3'
        accessToken = '1351343906-PJxiZ3Psqx6x2sgJ1vGNbZLgkqOtDKPJ6Bmphu9'
        accessTokenSecret = 'W9klp97oqQ40OjLEReERoXWkoRylajxn8jEZi5TKYRvUc'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth, wait_on_rate_limit=True)

        # input for term to be searched and how many tweets to search
        # searchTerm = input("Enter Keyword/Tag to search about: ")
        # NoOfTerms = int(input("Enter how many tweets to search: "))
        tweets = int(tweets)

        # searching for tweets
        self.tweets = tweepy.Cursor(
            api.search_tweets, q=keyword, lang="en").items(tweets)
        
        # Open/create a file to append data to
        csvFile = open('result.csv', 'w')

        # Use csv writer
        csvWriter = csv.writer(csvFile)

        # creating some variables to store info
        polarity = 0
        positive = 0
        wpositive = 0
        spositive = 0
        negative = 0
        wnegative = 0
        snegative = 0
        neutral = 0

        # iterating through tweets fetched
        for tweet in self.tweets:

            # Append to temp so that we can store in csv later. I use encode UTF-8

            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            original_tweet = tweet.text.encode('utf-8')
            recent_tweet = self.cleanTweet(tweet.text)

            # print (tweet.text.translate(non_bmp_map)) #print tweet's text
            analysis = TextBlob(tweet.text)

            recent_sentiment = analysis.sentiment.polarity

            # print(analysis.sentiment) # print tweet's polarity
            # adding up polarities to find the average later
            polarity += analysis.sentiment.polarity
            recent_label=''
            # adding reaction of how people are reacting to find average later
            if (analysis.sentiment.polarity == 0):
                neutral += 1
                recent_label ='neutral'
            elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
                wpositive += 1
                recent_label = 'weakly positive'
            elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                positive += 1
                recent_label = 'positive'
            elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
                spositive += 1
                recent_label = 'strongly positive'
            elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
                wnegative += 1
                recent_label = 'weakly negative'
            elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                negative += 1
                recent_label = 'negative'
            elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
                snegative += 1
                recent_label = 'strongly negative'

            check_tweet = pad_sequences(tokenizer.texts_to_sequences([recent_tweet]), maxlen=300)
            sent_score = gru_model.predict([check_tweet])[0]
            sent_label = ''
            if (sent_score > 0.50):
                sent_label = "Positive"
            elif (sent_score < 0.50):
                sent_label = "Negative"
            elif (sent_score==0.50):
                sent_label="Neutral"
            sent_score = np.round(sent_score*100,2)
            recent_row = [original_tweet,recent_tweet, recent_sentiment,recent_label,sent_score,sent_label]
            self.rows.append(recent_row)
        # Write to csv and close csv file
        fields =['Original_Tweet','Cleaned_Tweet','Polarity','Sentiment','Confidence','Predicted']
        csvWriter.writerow(fields)
        csvWriter.writerows(self.rows)
        csvFile.close()

        table = pd.read_csv('result.csv')
        table.to_html('templates/detail_table.html')
        # finding average of how people are reacting
        positive = self.percentage(positive, tweets)
        wpositive = self.percentage(wpositive, tweets)
        spositive = self.percentage(spositive, tweets)
        negative = self.percentage(negative, tweets)
        wnegative = self.percentage(wnegative, tweets)
        snegative = self.percentage(snegative, tweets)
        neutral = self.percentage(neutral, tweets)

        # finding average reaction
        polarity = polarity / tweets
        polarity = np.round(polarity,4)
        # printing out data
        # print("How people are reacting on " + keyword + " by analyzing " + str(tweets) + " tweets.")
        # print()
        # print("General Report: ")

        if (polarity == 0):
            htmlpolarity = "Neutral"

        # print("Neutral")
        elif (polarity > 0 and polarity <= 0.3):
            htmlpolarity = "Weakly Positive"
        # print("Weakly Positive")
        elif (polarity > 0.3 and polarity <= 0.6):
            htmlpolarity = "Positive"
        elif (polarity > 0.6 and polarity <= 1):
            htmlpolarity = "Strongly Positive"
        elif (polarity > -0.3 and polarity <= 0):
            htmlpolarity = "Weakly Negative"
        elif (polarity > -0.6 and polarity <= -0.3):
            htmlpolarity = "Negative"
        elif (polarity >= -1 and polarity <= -0.6):
            htmlpolarity = "strongly Negative"

        self.plotPieChart(positive, wpositive, spositive, negative,
                          wnegative, snegative, neutral, keyword, tweets)
        print(polarity, htmlpolarity)
        return polarity, htmlpolarity, positive, wpositive, spositive, negative, wnegative, snegative, neutral, keyword, tweets

    def cleanTweet(self, text):
        # Remove Links, Special Characters etc from tweet
        #return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())
        username = "@\S+"
        new_tweet = re.sub(username, ' ', text)  # Remove @tags
        new_tweet = new_tweet.lower()  # Smart lowercase
        new_tweet = re.sub(r'\d+', ' ', new_tweet)  # Remove numbers
        text_noise = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
        new_tweet = re.sub(text_noise, ' ', new_tweet)  # Remove links
        new_tweet = new_tweet.translate(new_tweet.maketrans('', '', string.punctuation))  # Remove Punctuation
        new_tweet = new_tweet.strip()  # Remove white spaces
        stop_words = set(stopwords.words('english'))
        new_tweet = ' '.join([word for word in new_tweet.split() if not word in stop_words])
        lemmatizer = WordNetLemmatizer()
        new_tweet = ' '.join([lemmatizer.lemmatize(word, "v") for word in new_tweet.split()])
        return new_tweet

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    # function which sets and plots the pie chart. The chart is saved in an img file every time the project is run.
    # The previous image is overwritten. This image is called in the html page.

    def plotPieChart(self, positive, wpositive, spositive, negative, wnegative, snegative, neutral, keyword, tweets):
        fig = plt.figure()
        labels = ['Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]',
                  'Strongly Positive [' + str(spositive) +
                  '%]', 'Neutral [' + str(neutral) + '%]',
                  'Negative [' + str(negative) +
                  '%]', 'Weakly Negative [' + str(wnegative) + '%]',
                  'Strongly Negative [' + str(snegative) + '%]']
        sizes = [positive, wpositive, spositive,
                 neutral, negative, wnegative, snegative]
        colors = ['yellowgreen', 'lightgreen', 'darkgreen',
                  'gold', 'red', 'lightsalmon', 'darkred']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.axis('equal')
        plt.tight_layout()
        strFile = r"C:\Users\91700\Desktop\offensivetweets\static\images\piee.png"
        if os.path.isfile(strFile):
            os.remove(strFile)  # Opt.: os.system("rm "+strFile)
        plt.savefig(strFile)
        plt.show()


@second.route('/sentiment_logic', methods=['POST', 'GET'])
def sentiment_logic():
    # get user input of keyword to search and number of tweets from html form.
    keyword = request.form.get('keyword')
    tweets = request.form.get('tweets')
    sa = SentimentAnalysis()

    # set variables which can be used in the jinja supported html page
    polarity, htmlpolarity, positive, wpositive, spositive, negative, wnegative, snegative, neutral, keyword1, tweet1 = sa.DownloadData(
        keyword, tweets)
    return render_template('sentiment_analyzer.html', polarity=polarity, htmlpolarity=htmlpolarity, positive=positive,
                           wpositive=wpositive, spositive=spositive,
                           negative=negative, wnegative=wnegative, snegative=snegative, neutral=neutral,
                           keyword=keyword1, tweets=tweet1)

@second.route("/offensive_analyzer")
def offensive_analyzer():
    return render_template("offensive_analyzer.html")

class OffenseAnalysis:

    def __init__(self):
        self.tweet_list = []
        self.tweet_rows = []

    def repair(self,text):
        username = "@\S+"
        new_tweet = re.sub(username, ' ', text)  # Remove @tags
        new_tweet = new_tweet.lower()  # Smart lowercase
        new_tweet = re.sub(r'\d+', ' ', new_tweet)  # Remove numbers
        text_noise = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
        new_tweet = re.sub(text_noise, ' ', new_tweet)  # Remove links
        new_tweet = new_tweet.translate(new_tweet.maketrans('', '', string.punctuation))  # Remove Punctuation
        new_tweet = new_tweet.strip()  # Remove white spaces
        stop_words = set(stopwords.words('english'))
        new_tweet = ' '.join([word for word in new_tweet.split() if not word in stop_words])
        lemmatizer = WordNetLemmatizer()
        new_tweet = ' '.join([lemmatizer.lemmatize(word, "v") for word in new_tweet.split()])
        return new_tweet

    def predict(self,text,model,model_name):
        # Tokenize text
        cleaned_text = self.repair(text)
        x_test = pad_sequences(tokenizer.texts_to_sequences([cleaned_text]), maxlen=max_sequence_length)
        # Predict
        score = model.predict([x_test])[0]
        confidence = 0
        label=''
        unlabel=''
        if (score >= 0.50):
            label = "Positive"
            unlabel = "Negative"
            confidence = np.round(score*100,2)
        elif (score < 0.50):
            label = "Negative"
            unlabel = "Positive"
            score = 1-score
            confidence = np.round(score * 100, 2)
        unconfidence = np.round((100-confidence),2)

        recent_row = [text, cleaned_text, model_name, label, confidence[0]]
        self.tweet_rows.append(recent_row)
        # Write to csv and close csv file
        csvFile = open('tweet_list.csv', 'a')
        csvWriter = csv.writer(csvFile)
       # fields = ['Tweet', 'Cleaned_Tweet', 'Model', 'Prediction', 'Confidence']
       # csvWriter.writerow(fields)
        csvWriter.writerows(self.tweet_rows)
        #csvFile.close()

        return label,unlabel,confidence[0],unconfidence[0],cleaned_text

@second.route('/offensive_logic', methods=['POST', 'GET'])
def offensive_logic():
    # get user input of keyword to search and number of tweets from html form.
    tweet_text = request.form.get('tweet_text')
    model_value = request.form.get('models')
    model_name = ''
    if model_value == 'cnn':
        model = cnn_model
        model_name = 'Basic CNN'
    elif model_value == 'gru':
        model = gru_model
        model_name = 'CNN + Bidirectional GRU'
    else:
        model = lstm_model
        model_name = 'Bidirectional LSTM'

    oa = OffenseAnalysis()
    # set variables which can be used in the jinja supported html page
    sentiment,oppsentiment,confi,unconfi,repaired_text = oa.predict(tweet_text,model,model_name)
    return render_template('offensive_analyzer.html',your_tweet=tweet_text,repaired=repaired_text,your_model=model_name, tweet_sentiment=sentiment,opp_sentiment=oppsentiment, confidence_score = confi,unconfidence_score = unconfi)




@second.route("/compare_analyzer")
def compare_analyzer():
    return render_template("compare_sent.html")

class CompareAnalysis:

    def __init__(self):
        self.tweet_list = []
        self.tweet_rows = []

    def repair(self,text):
        username = "@\S+"
        new_tweet = re.sub(username, ' ', text)  # Remove @tags
        new_tweet = new_tweet.lower()  # Smart lowercase
        new_tweet = re.sub(r'\d+', ' ', new_tweet)  # Remove numbers
        text_noise = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
        new_tweet = re.sub(text_noise, ' ', new_tweet)  # Remove links
        new_tweet = new_tweet.translate(new_tweet.maketrans('', '', string.punctuation))  # Remove Punctuation
        new_tweet = new_tweet.strip()  # Remove white spaces
        stop_words = set(stopwords.words('english'))
        new_tweet = ' '.join([word for word in new_tweet.split() if not word in stop_words])
        lemmatizer = WordNetLemmatizer()
        new_tweet = ' '.join([lemmatizer.lemmatize(word, "v") for word in new_tweet.split()])
        return new_tweet

    def predict(self,text):
        # Tokenize text
        cleaned_text = self.repair(text)
        x_test = pad_sequences(tokenizer.texts_to_sequences([cleaned_text]), maxlen=max_sequence_length)
        # Predict
        score_cnn = cnn_model.predict([x_test])[0]
        score_gru = gru_model.predict([x_test])[0]
        score_lstm = lstm_model.predict([x_test])[0]


        confidence_cnn = 0
        label_cnn=''
        unlabel_cnn=''
        if (score_cnn >= 0.50):
            label_cnn = "Positive"
            unlabel_cnn = "Negative"
            confidence_cnn = np.round(score_cnn*100,2)
        elif (score_cnn < 0.50):
            label_cnn = "Negative"
            unlabel_cnn = "Positive"
            score_cnn = 1-score_cnn
            confidence_cnn = np.round(score_cnn * 100, 2)
        unconfidence_cnn = np.round((100-confidence_cnn),2)

        confidence_gru = 0
        label_gru = ''
        unlabel_gru = ''
        if (score_gru >= 0.50):
            label_gru = "Positive"
            unlabel_gru = "Negative"
            confidence_gru = np.round(score_gru * 100, 2)
        elif (score_gru < 0.50):
            label_gru = "Negative"
            unlabel_gru = "Positive"
            score_gru = 1 - score_gru
            confidence_gru = np.round(score_gru * 100, 2)
        unconfidence_gru = np.round((100 - confidence_gru), 2)

        confidence_lstm = 0
        label_lstm = ''
        unlabel_lstm = ''
        if (score_lstm >= 0.50):
            label_lstm = "Positive"
            unlabel_lstm= "Negative"
            confidence_lstm = np.round(score_lstm * 100, 2)
        elif (score_lstm < 0.50):
            label_lstm = "Negative"
            unlabel_lstm = "Positive"
            score_lstm = 1 - score_lstm
            confidence_lstm = np.round(score_lstm * 100, 2)
        unconfidence_lstm = np.round((100 - confidence_lstm), 2)
        #recent_row = [text, cleaned_text, model_name, label, confidence[0]]
        #self.tweet_rows.append(recent_row)
        # Write to csv and close csv file
        #csvFile = open('tweet_list.csv', 'a')
        #csvWriter = csv.writer(csvFile)
       # fields = ['Tweet', 'Cleaned_Tweet', 'Model', 'Prediction', 'Confidence']
       # csvWriter.writerow(fields)
        #csvWriter.writerows(self.tweet_rows)
        #csvFile.close()

        return label_cnn,label_gru,label_lstm,unlabel_cnn,unlabel_gru,unlabel_lstm,confidence_cnn[0],confidence_gru[0],confidence_lstm[0],unconfidence_cnn[0],unconfidence_gru[0],unconfidence_lstm[0],cleaned_text

@second.route('/compare_logic', methods=['POST', 'GET'])
def compare_logic():
    # get user input of keyword to search and number of tweets from html form.
    com_tweet = request.form.get('com_tweet')
    ca = CompareAnalysis()
    # set variables which can be used in the jinja supported html page
    cnn_sentiment,gru_sentiment,lstm_sentiment,cnn_oppsentiment,gru_oppsentiment,lstm_oppsentiment,cnn_confi,gru_confi,lstm_confi,cnn_unconfi,gru_unconfi,lstm_unconfi,repaired_text = ca.predict(com_tweet)

    return render_template('compare_sent.html',your_tweet=com_tweet,repaired=repaired_text, tweet_sentiment_cnn=cnn_sentiment,tweet_sentiment_gru=gru_sentiment,tweet_sentiment_lstm=lstm_sentiment,opp_sentiment_cnn=cnn_oppsentiment,opp_sentiment_gru=gru_oppsentiment,opp_sentiment_lstm=lstm_oppsentiment, confidence_score_cnn = cnn_confi,confidence_score_gru = gru_confi,confidence_score_lstm = lstm_confi,unconfidence_score_cnn = cnn_unconfi,unconfidence_score_gru = gru_unconfi,unconfidence_score_lstm = lstm_unconfi)


@second.route('/visualize')
def visualize():
    return render_template('PieChart.html')

@second.route('/analysis_logic', methods=['POST', 'GET'])
def analysis_logic():
    # get user input of keyword to search and number of tweets from html form.
    preds = request.form.get('pred')
    recent_row = [preds]
    csvFile = open('pred_list.csv', 'a')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(recent_row)
    tweet_data = pd.read_csv('tweet_list.csv')
    #pred_data = pd.read_csv('pred_list.csv')
    return render_template('PieChart.html',prediction=preds)

@second.route('/display_table')
def display_table():
    return render_template('detail_table.html')

@second.route('/cnn_det')
def cnn_det():
    return render_template('cnn.html')

#@second.route('/check_cnn', methods=['POST', 'GET'])
#def check_cnn():
    # get user input of keyword to search and number of tweets from html form.
    #cn_data = pd.read_csv('tweet_list.csv')
    #cn_data = cn_data.dropna()

    #return render_template('compare_sent.html',your_tweet=com_tweet,repaired=repaired_text, tweet_sentiment_cnn=cnn_sentiment,tweet_sentiment_gru=gru_sentiment,tweet_sentiment_lstm=lstm_sentiment,opp_sentiment_cnn=cnn_oppsentiment,opp_sentiment_gru=gru_oppsentiment,opp_sentiment_lstm=lstm_oppsentiment, confidence_score_cnn = cnn_confi,confidence_score_gru = gru_confi,confidence_score_lstm = lstm_confi,unconfidence_score_cnn = cnn_unconfi,unconfidence_score_gru = gru_unconfi,unconfidence_score_lstm = lstm_unconfi)


@second.route('/gru_det')
def gru_det():
    return render_template('gru.html')

@second.route('/lstm_det')
def lstm_det():
    return render_template('lstm.html')