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

def clean_string(inp):
	return "".join(i if 0 <= ord(i) < 128 else " " for i in inp);

a=set();

def splitter(inp):
	global a;
	inp = re.sub("(https?://[^ ]*)|([@][a-zA-Z0-9_]+)|(&[a-zA-Z0-9]+;)", "", inp.lower()); #Remove URL, @<Username>, &nbsp;
	tokens = tknzr.tokenize(clean_string(inp));
	tokens = nltk.sentiment.util.mark_negation(tokens)
	output = defaultdict(int)
	for i in tokens:
		synsets = sentiwordnet.senti_synsets(i)
		scores = list(x.pos_score() - x.neg_score() for x in synsets)
		thisscore = sum(scores)/float(len(scores)) if (len(scores) >  0) else 0
		if(thisscore > 0):
			output["MS_POS"]+=1;
		if(thisscore < 0):
			output["MS_POS"]-=1;
		if(i in ['#india', '#government', '#rss', '#primeminister', '#narendramodi', '#bjp', '#modi', '#namo']):
			output[i]+=1;
		elif(i[0] == "#"):
			output[i]+=2;
		else:
			output[i]+=1;

#[('#aap', 2), ('#delhi', 2), ('#bihar', 2), ('#cleanindia', 2), ('#netneutrality', 2), ('#narendra', 2), ('#feku', 2), ('#india', 2), ('#hindutva', 2), ('#internationalfleetreview', 2), ('#innovation', 2), ('#press', 2), ('#government', 2), ('#nuclear', 2), ('#makeinindia', 2), ('#landbill', 2), ('#environment', 2), ('#bhakts', 2), ('#development', 2), ('#news', 2), ('#global', 2), ('#rss', 2), ('#cong', 2), ('#hanumanthappa', 2), ('#sensex', 2), ('#siachenavalanche', 2), ('#arunjaitley', 2), ('#startups', 2), ('#progress', 2), ('#management', 2), ('#media', 2), ('#acchedin', 2), ('#primeminister', 2), ('#mycleanindia', 3), ('#narendramodi', 3), ('#startupindia', 3), ('#raga', 3), ('#bjp', 3), ('#modi', 4), ('#namo', 5)]

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
	data = [];
	datatest = [];

	targetcsv = read_csv('modi_train.csv', True);

	print "Done the shuffle";
	count = 1;
	for row in targetcsv:
		wordbag = splitter(row[1]);
		if(count <= len(targetcsv) - 600 ):
			datatest.append((row[0], row[1], wordbag))
		else:
			data.append((row[0], row[1], wordbag));
		count += 1;
	print "Done Spliting Data!";

	CV = sklearn.feature_extraction.DictVectorizer();
	X = CV.fit_transform(list(i[2] for i in data))
	y = np.array(list( i[0] for i in data));
	if(dumpname != None):
		f = open("CV"+dumpname+".ms", "wb");
		pickle.dump(CV, f);
		f.close();

	clf = linear_model.LogisticRegression();
	clf.fit(X, y);
	print "Learnt !... Ready to save in a file";
	if(dumpname != None):
		dump_learnt = open(dumpname+".ms", "wb");
		pickle.dump(clf, dump_learnt);
		dump_learnt.close();

	X1 = CV.transform(list(i[2] for i in datatest))
	y1 = clf.predict(X1);

	correct = sum((y1[i] == datatest[i][0]) for i in xrange(len(datatest)));
	print "correct = ", correct, "datatest = ", len(datatest);
	print "Accuracy = ", (correct*100.0)/len(datatest);
mainfunc();

