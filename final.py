import pickle
import sys;
import numpy as np
import sklearn
from sklearn.feature_extraction import *
from sklearn import linear_model
from nltk.tokenize import TweetTokenizer
from nltk.sentiment import *
import csv,re,random, math
import nltk;
_runmode = {"not": 1};
tknzr = TweetTokenizer();
def clean_string(inp):
	return "".join(i if 0 <= ord(i) < 128 else " " for i in inp);
def splitter(inp):
	inp = re.sub("(https?://[^ ]*)|([@][a-zA-Z0-9_]+)|(&[a-zA-Z0-9]+;)", "", inp.lower()); #Remove URL, @<Username>, &nbsp;
	t_tokens = tknzr.tokenize(clean_string(inp));
	tokens = t_tokens;
	tokens = nltk.sentiment.util.mark_negation(t_tokens)
	return tokens;
def mainfunc():
	f = open(sys.argv[1]);
	datatest = f.read().strip().split("\n");
	f.close();
	f = open( "CValgo_fulldata.ms", "r");
	CV = pickle.load(f);
	f.close();
	load_learnt = open("algo_fulldata2.ms", "r");
	clf = pickle.load(load_learnt);
	load_learnt.close();
	X1 = CV.transform(list(" ".join(splitter(i)) for i in datatest));
	y1 = clf.predict(X1);
	f=open(sys.argv[2], "w");
	for i in y1:
		f.write(i+"\n");
	f.close();
mainfunc();
