import tweepy as tweepy
from tkinter import *
from tkinter import ttk, messagebox
from tweepy import API 
from tweepy import OAuthHandler
from tweepy import Cursor
from retrying import retry
import threading
import time
from pandas import read_csv
import numpy
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os




class TrainingWindow:    
    def getTrainingSet(self):
            from tkinter.filedialog import askopenfilename

            Tk().withdraw() 
            self.trainingcsv = askopenfilename(filetypes = (("CSV files","*.csv"),("all files","*.*")))
            val = re.search(r".csv", self.trainingcsv)
            if val: 
                self.entry_1.insert(0, str(self.trainingcsv))
                print("[+] Training dataset is: " + str(self.trainingcsv))
                return
            self.trainingcsv = ""
            messagebox.showinfo("Error", "Valid datasets must be in CSV format")
            return


    def trainWithDataset(self):
        self.NUM_EPOCHS = int(self.entry_4.get())
        ftrain = open(self.trainingcsv, "r", encoding='utf8')
        names = ['tweetid', 'sentiment', 'tweet']
        data = read_csv(ftrain, names=names)
        #lines = csvfile.read()
        total = data['tweetid'].count()
        print("Total is: " + str(total))
        for i in range(total - 1):
            label = data['sentiment'][i]
            tweet = data['tweet'][i].lower()
            words = nltk.word_tokenize(tweet)
            if len(words) > self.maxlen:
                self.maxlen = len(words)
            for word in words:
                self.word_freqs[word] += 1
            self.num_recs += 1
        ftrain.close()

        vocab_size = min(self.MAX_FEATURES, len(self.word_freqs)) + 2
        word2index = {x[0]: i+2 for i, x in
        enumerate(self.word_freqs.most_common(self.MAX_FEATURES))}
        word2index["PAD"] = 0
        word2index["UNK"] = 1
        index2word = {v:k for k, v in word2index.items()}

        X = np.empty((self.num_recs, ), dtype=list)
        y = np.zeros((self.num_recs, ))
        i = 0
        ftrain = open(self.trainingcsv, 'r')
        names = ['tweetid', 'sentiment', 'tweet']
        data = read_csv(ftrain, names=names)
        total = data['tweetid'].count()
        for i in range(total - 1):
            label = data['sentiment'][i]
            sentence = data['tweet'][i].lower()
            words = nltk.word_tokenize(sentence)
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            y[i] = int(label)
            i += 1
        ftrain.close()
        X = sequence.pad_sequences(X, maxlen=self.MAX_SENTENCE_LENGTH)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Embedding(vocab_size, self.EMBEDDING_SIZE,
        input_length=self.MAX_SENTENCE_LENGTH))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(self.HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam",
            metrics=["accuracy"])

        history = model.fit(Xtrain, ytrain, batch_size=self.BATCH_SIZE, epochs=self.NUM_EPOCHS,
            validation_data=(Xtest, ytest))

        #evaluate model
        score, acc = model.evaluate(Xtest, ytest, batch_size=self.BATCH_SIZE)
        report = "Model Trained Successfuly. Test Score: " + str(score) + ". Accuracy is: " + str(acc*100) + "%"
        messagebox.showinfo("Training Report", report)
        print("Test score: %.3f, accuracy: %.3f" % (score, acc))
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        self.progress_bar["value"] = 100
        print("Model saved")

        for i in range(5):
            idx = np.random.randint(len(Xtest))
            xtest = Xtest[idx].reshape(1,40)
            ylabel = ytest[idx]
            ypred = model.predict(xtest)[0][0]
            sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
            print("%.0ft%dt%s" % (ypred, ylabel, sent))
            
  
    def __init__(self, master):
        frame = Frame(master, width=500, height=500)
        frame.pack_propagate(0)
        frame.pack(fill=BOTH, expand=1)

        self.DATA_DIR = r"C:\Users\Kev\Documents\Code\tkinter"

        self.MAX_FEATURES = 2000
        self.MAX_SENTENCE_LENGTH = 40

        self.EMBEDDING_SIZE = 128
        self.HIDDEN_LAYER_SIZE = 64
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 1

        self.maxlen = 0
        self.word_freqs = collections.Counter()
        self.num_recs = 0
        self.stopWords = ['AT_USER', 'URL']
        
        self.tweetidsTrain = list()
        self.tweetsetTrain = list()
        self.labelsTrain = list()
        self.tweetidsTest = list()
        self.tweetsetTest = list()
        self.labelsTest = list()
        self.thingname = "Training Window"
        self.label_1 = Label(frame, text="Select a Dataset to train your Model ", width=50, height=5, font="Helvetica 15 bold")
        self.label_3 = Label(frame, text="Number of Epochs ")
        self.entry_1 = Entry(frame)
        self.entry_4 = Entry(frame, width=5)
        self.button_1 = Button(frame, text="File", command=self.getTrainingSet)
        self.button_3 = Button(frame, text="Train ML model", command=self.trainWithDataset)
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=100)
        self.progress_bar.config(mode='determinate', maximum=100, value=0)
        self.progress_bar_2 = ttk.Progressbar(frame, orient="horizontal", length=100)
        
        self.label_1.pack()
        self.entry_1.pack()
        self.button_1.pack()
        self.label_3.pack()
        self.entry_4.pack()
        self.button_3.pack()
        self.progress_bar.pack()

