from nltk.corpus import sentiwordnet
from collections import defaultdict
import numpy as np
import sklearn
from sklearn.feature_extraction import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import linear_model
from nltk.tokenize import TweetTokenizer
from nltk.sentiment import *
import csv,re,random, math, nltk, pickle


random.seed(266);

tknzr = TweetTokenizer();


tokens = tknzr.tokenize("I don't like you :/ I just hate you");

tokens = nltk.sentiment.util.mark_negation(tokens)
print sum(x[-4:] == "_NEG" for x in tokens);

print tokens;

