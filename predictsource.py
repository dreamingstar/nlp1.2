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

def predict(clf, x):
	return list(("0" if (i[0]>=57) else ( "4" if i[0] <= 43 else "2")) for i in clf.predict_proba(x));

def clean_string(inp):
	return "".join(i if 0 <= ord(i) < 128 else " " for i in inp);

def splitter(inp):
	inp = re.sub("(https?://[^ ]*)|([@][a-zA-Z0-9_]+)|(&[a-zA-Z0-9]+;)", "", inp.lower()); #Remove URL, @<Username>, &nbsp;
	tokens = tknzr.tokenize(clean_string(inp));
	tokens = nltk.sentiment.util.mark_negation(tokens)
	return tokens;

dumpname = "algo_15l_7march";

def read_csv(fn, shuffled = False):
	csvfd = open(fn, 'rb');
	sp = csv.reader(csvfd)
	csvdata =  list(row for row in sp);
	csvfd.close();
	if(shuffled):
		random.shuffle(csvdata);
	return csvdata;

def mainfunc():
	datatest = [];
	sourcecsv = read_csv('../training.csv', True);
	print "Done the shuffle";
	count = 1;
	for row in sourcecsv:
		wordbag = splitter(row[1]);
		if(count <= (len(sourcecsv)*10)/100):
			datatest.append((row[0], row[1], wordbag))
		else:
			break;
		count += 1;

	print "Parsed Datatest";

	f = open("CV"+dumpname+".ms", "r");
	CV = pickle.load(f);
	f.close();

	load_learnt = open( dumpname+".ms", "r" );
	clf = pickle.load( load_learnt );
	load_learnt.close();

	X1 = CV.transform(list(" ".join(i[2]) for i in datatest))
	y1 = clf.predict(X1);

	correct = sum((y1[i] == datatest[i][0]) for i in xrange(len(datatest)));
	print "correct = ", correct, "datatest = ", len(datatest);
	if(True):
		print "Accuracy = ", (correct*100.0)/len(datatest);
	else:
		print "Accuracy = ", (correct*100.0)/sum((i[0] in ["0", "4"]) for i in datatest);


mainfunc();

