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

def predict(clf, x):
	return list(("0" if (i[0]>=57) else ( "4" if i[0] <= 43 else "2")) for i in clf.predict_proba(x));

def predict_modi(clf, X1, wordbags):
	y1 = clf.predict(X1);
	return y1;
	for i in xrange(len(wordbags)):
		for j in wordbags[i]:
			if(j[0] == "#"):
				if(re.search("bhakt", j) != None):
					y1[i] = '0';
				if(re.search("aaptard", j) != None or re.search("aapians", j) != None):
					y1[i] = '4';
	return y1;

def clean_string(inp):
	return "".join(i if 0 <= ord(i) < 128 else " " for i in inp);

def splitter(inp):
	inp = re.sub("(https?://[^ ]*)|(&[a-zA-Z0-9]+;)", " ", inp); #Remove URL, @<Username>, &nbsp;
	inp = re.sub("([@][a-zA-Z0-9_]+)", " @person ", inp);
	tokens = tknzr.tokenize(clean_string(inp));
	tokens = nltk.sentiment.util.mark_negation(tokens)
	output = defaultdict(int)

	etokens = [];
	for i in tokens:
		if(i[0] == "#"):
			etokens += re.sub( r"([A-Z]+[a-z0-9]+)", r" \1", i[1:]).split();
	tokens += etokens;
	for i in tokens:
		if(i.isupper() or i[0].isupper()):
			output[i]+=1;
		i=i.lower();
		synsets = sentiwordnet.senti_synsets(i)
		scores = list(x.pos_score() - x.neg_score() for x in synsets)
		thisscore = sum(scores)/float(len(scores)) if (len(scores) >  0) else 0
		if(thisscore > 0):
			output["MS_POS"]+=1;
		if(thisscore < 0):
			output["MS_NEG"]+=1;
		if(i[0] == "#"):
			# if(ht_replace.has_key(i)):
			# 	i = "#"+ht_replace[i];
			# if(i in good_words):
			# 	output["MS_POSS"]+=1;
			# if(i in bad_words):
			# 	output["MS_NEGG"]+=1;
			if(re.search("bhakt", i) != None):
				output["MS_NEG"] += 5;
			if(re.search("aaptard", i) != None or re.search("aapians", i) != None):
				output["MS_POS"] += 5;
			output[i]+=2;
		else:
			output[i]+=1;
	return output;


dumpname = "algo_600";

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

	targetcsv = read_csv('modi_train.csv', True);


	print "Done the shuffle";
	count = 1;
	for row in targetcsv:
		wordbag = splitter(row[1]);
		if(count <= len(targetcsv) - 600 ):
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

	X1 = CV.transform(list(i[2] for i in datatest))
	y1 = clf.predict(X1);

	correct = sum((y1[i] == datatest[i][0]) for i in xrange(len(datatest)));
	print "correct = ", correct, "datatest = ", len(datatest);
	if(True):
		print "Accuracy = ", (correct*100.0)/len(datatest);
	else:
		print "Accuracy = ", (correct*100.0)/sum((i[0] in ["0", "4"]) for i in datatest);
mainfunc();
