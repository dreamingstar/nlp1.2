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

def splitter(inp):
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
			output["MS_NEG"]+=1;
		if(i[0] == "#"):
			output["MS_hashtag"]+=2;
			i = i[1:];
		output[i]+=1;
	return output;

dumpname = "algo_15l_7march1";

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

	sourcecsv = read_csv('../training.csv', True);

	#sourcecsv = sourcecsv[:50000]

	print "Done the shuffle";
	count = 1;
	for row in sourcecsv:
		wordbag = splitter(row[1]);
		if(count <= (len(sourcecsv)*10)/100):
			datatest.append((row[0], row[1], wordbag))
		else:
			data.append((row[0], row[1], wordbag));
		if(count % 10000 == 0):
			print count;
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

