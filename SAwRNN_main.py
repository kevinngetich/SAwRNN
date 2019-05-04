import tweepy as tweepy
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tweepy import API 
from tweepy import OAuthHandler
from tweepy import Cursor
from retrying import retry
import threading
import time
import tweepy as tweepy
from pandas import read_csv
import numpy
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.models import load_model
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import random

from traininggui import *
import twitter_credentials

count = 0

def openTrainingWindow():
    root2 = Toplevel(root)
    root2.geometry("500x350")
    wintrain = TrainingWindow(root2)

def updateProgressBar(thing):
    global count
    while (count < 1001):

        time.sleep(1)
        print("PB Value for " + str(thing.thingname))
        count += 1


class FetchThread (threading.Thread):
   def __init__(self, threadID, name, object):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.thing = object
      self.tweetset = []

   def run(self):
      print ("Fetching Tweets: " + str(self.thing.entry_1))
      print("Fetching Tweets: " + str(self.thing.v))
      print("Fetching Tweets")
      self.tweetset = self.thing.twitter_client.get_tweets(self.thing.entry_1, self.thing.v, self.thing)
      self.thing.tweetset = self.tweetset
      

class TwitterClient():

    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user
        self.api = API(self.auth)



    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def get_tweets(self, search_query, max_tweets, object):
        print("[+] Starting..." + str(search_query) + " x " + str(max_tweets))
        global count
        tweet_text_set = []
        text = "Search Parameters" + str(search_query) + str(max_tweets)
        try:
           for tweet in Cursor(self.api.search, q=search_query).items(max_tweets):
                tweet_text_set.append(tweet.text)
                print("[+] Appending Tweets to created set")
                count += 1
                print("[+] 1 new append")
                percent = int(round(count/max_tweets * 100))
                #object.progress_bar.step()
                print("New Count: " + str(count) + ". Progress: " + str(percent) + "%")
        except tweepy.error.TweepError:
            print("[!] Connection Error. Please Check Your Internet Connection")
            raise

        return tweet_text_set


class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class InputWindow:

    def __init__(self, master, height=300, width=300):
        frame = Frame(master)
        frame.pack()


        self.thingname = "Main Window"
        self.master = master
        self.tweetset = []
        self.positivepredictions = []
        self.negativepredictions = []
        self.numpositives = 0
        self.numnegatives = 0
        self.totalanalyzed = 0
        self.MAX_FEATURES = 2000
        self.MAX_SENTENCE_LENGTH = 40

        self.EMBEDDING_SIZE = 128
        self.HIDDEN_LAYER_SIZE = 64
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 1

        self.maxlen = 0
        self.word_freqs = collections.Counter()
        self.num_recs = 0
        self.label_1 = Label(frame, text="Welcome to SAwRNN ", width=25, height=3, font="Helvetica 15 bold")
        self.label_2 = Label(frame, text="Search Topic ")
        self.label_3 = Label(frame, text="Max Number of Tweets to Analyse")
        self.entry_1 = Entry(frame)
        self.button_1 = Button(frame, text="GO", command=self.getTweets)
        self.button_2 = Button(frame, text="Analysis", command=self.getAnalysis)
        MAX = [
            ("100 ", 100),
            ("1000  ", 1000),
            ("100,000", 100000),
            ("1,000,000", 1000000),
        ]
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=100)
        self.progress_bar.config(mode='determinate', maximum=100, value=0)


        self.v = IntVar()
        self.v.set(100)  # initialize


        self.label_1.grid(row=0, column=2)
        self.label_2.grid(row=1, column=2)
        self.entry_1.grid(row=2, column=2)
        self.label_3.grid(row=3, column=2)
        r = 3
        for text, m in MAX:
            r = r + 1
            self.b = Radiobutton(frame, text=text,
                            variable=self.v, value=m)
            self.b.grid(row=r, column=2)
        self.button_1.grid(row=8, column=2)
        self.progress_bar.grid(row=9, column=2)
        self.button_2.grid(row=10, column=2)



    twitter_client = TwitterClient()


    def getTweets(self):
        #_thread.start_new_thread(updateProgressBar(self), ("Thread-1", 2,))
        #_thread.start_new_thread(self.twitter_client.get_tweets(self.entry_1.get(), self.v.get(), self), ("Thread-2", 4,))

        text = self.entry_1.get()
        self.entry_1 = self.entry_1.get()
        self.v = self.v.get()
        if len(text) > 1:
            thread2 = FetchThread(2, "Fetch Thread", self)

            # Start new Threads
            thread2.start()
            thread2.join()
            self.progress_bar["value"] = 100
            print("done getting: " + self.entry_1)
            sampletweet = self.tweetset[5]
            char_list = [sampletweet[j] for j in range(len(sampletweet)) if ord(sampletweet[j]) in range(65536)]
            sampletweet=''
            for j in char_list:
                sampletweet=sampletweet+j
            messagebox.showinfo("Sample Tweet", sampletweet)
            return
        messagebox.showinfo("Error", "Search Field Cannot Be Empty")
        return



    def getAnalysis(self):
        for tweet in self.tweetset:
            #print(tweet)
            tweet = tweet.lower()
            words = nltk.word_tokenize(tweet)
            if len(words) > self.maxlen:
                self.maxlen = len(words)
            for word in words:
                self.word_freqs[word] += 1
            self.num_recs += 1

        vocab_size = min(self.MAX_FEATURES, len(self.word_freqs)) + 2
        word2index = {x[0]: i+2 for i, x in
        enumerate(self.word_freqs.most_common(self.MAX_FEATURES))}
        word2index["PAD"] = 0
        word2index["UNK"] = 1
        index2word = {v:k for k, v in word2index.items()}

        X = np.empty((self.num_recs, ), dtype=list)
        i = 0
        for tweet in self.tweetset:
            sentence = tweet.lower()
            words = nltk.word_tokenize(sentence)
            seqs = []
            #print("For tweet in tweetset, tweet is: " + str(sentence))
            for word in words:
                #print("For Word in words, word is: " + str(word))
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            #print(seqs)
            i += 1
        
        X = sequence.pad_sequences(X, maxlen=self.MAX_SENTENCE_LENGTH)
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        for i in range(len(X)):
            xtest = X[i].reshape(1,40)
            #print("xtest: " + str(xtest))
            ypred = loaded_model.predict(xtest)[0][0]
            sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
            print("Prediction: " + str(i) + " : " + str(ypred))
            if ypred > 0.7:
                self.numpositives += 1
                self.positivepredictions.append(sent)
            else:
                self.numnegatives += 1
                self.negativepredictions.append(sent)
        predict = "Positive: " + str(self.numpositives) + " Negative: " + str(self.numnegatives)
        self.totalanalyzed = self.numpositives + self.numnegatives
        #messagebox.showinfo("Done", predict)
        print("Done Making predictions")            
        
        
        results = Toplevel(self.master)
        results.geometry("1000x500")
        results.totaltweetsanalyzed = self.totalanalyzed
        results.totalpositivetweets = self.numpositives
        results.totalnegativetweets = self.numnegatives
        results.pos = []
        results.neg = []
        results.overallresult = ""
        results.searchtopic = self.entry_1
        if results.totalpositivetweets > results.totalnegativetweets:
            results.overallresult = "Positive"
        else:
            results.overallresult = "Negative"

        values = []
        values.append(results.totalpositivetweets)
        values.append(results.totalnegativetweets)
        lab = ['Positive Tweets','Negative Tweets']
        labl = list(lab)
        cols = ['c','m']

        oaresult = "Overall public opinion as regards " + results.searchtopic + " is: " + results.overallresult
        analyzed = "Number of analyzed tweets: " + str(results.totaltweetsanalyzed)
        positive = "Number of positive tweets: " + str(results.totalpositivetweets)
        negative = "Number of negative tweets: " + str(results.totalnegativetweets)

        results.thingname = "Result Window"
        results.label_1 = Label(results, text="Analysis Report", width=50, height=3, font="Helvetica 20 bold")
        results.label_2 = Label(results, text=oaresult)
        results.label_3 = Label(results, text=analyzed)
        results.label_4 = Label(results, text=positive)
        results.label_5 = Label(results, text=negative)
        results.label_6 = Label(results, text="Sample Positive Tweets")
        results.label_7 = Label(results, text="-------------------------------------")
        results.label_8 = Label(results, text="Sample Negative Tweets")
        results.label_9 = Label(results, text="-------------------------------------")
        
        poscount = 0
        negcount = 0
        
        
        results.label_1.pack()
        results.label_2.pack()
        results.label_3.pack()
        results.label_4.pack()
        results.label_5.pack()
        results.label_7.pack()
        results.label_6.pack()
        while poscount < 3:
            results.p = Label(results,
                                          text=str(self.positivepredictions[random.randint(1, results.totalpositivetweets - 1)])
                                        )
            results.p.pack()
            poscount += 1
        results.label_9.pack()
        results.label_8.pack()
        while negcount < 3:
            results.n = Label(results,
                                          text=str(self.negativepredictions[random.randint(1, results.totalnegativetweets - 1)])
                                          )
            results.n.pack()
            negcount +=1

        plt.pie(values, labels=lab, colors=cols)
        plt.title('Data Summary')
        plt.legend()
        plt.show()
   





root = Tk()
root.title("SAwRNN - Sentiment Analysis Application")
root.geometry("500x350")
menu = Menu(root)
root.config(menu=menu)
subMenu = Menu(menu)
menu.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="New Project")
subMenu.add_command(label="Exit")

netMenu = Menu(menu)
menu.add_cascade(label="Network", menu=netMenu)
netMenu.add_command(label="Train Network", command=openTrainingWindow)


win = InputWindow(root)
root.mainloop()


entry_2 = Entry(root)
c = Checkbutton(root, text="Keep me logged in")
