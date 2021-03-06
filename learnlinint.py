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

def neg_occurence(tokens):
	outp = 0;
	inseq = False;
	for i in tokens:
		if(i[-4:] == "_NEG"):
			if(not(inseq)):
				outp+=1;
			inseq = True;
		else:
			inseq = False;
	return outp;

def splitter(inp):
	global a;
	inp = re.sub("(https?://[^ ]*)|(&[a-zA-Z0-9]+;)", " ", inp); #Remove URL, @<Username>, &nbsp;
	inp = re.sub("([@][a-zA-Z0-9_]+)", " @person ", inp);
	tokens = tknzr.tokenize(clean_string(inp));
	tokens = nltk.sentiment.util.mark_negation(tokens)
	output = defaultdict(int)
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
			output[i]+=2;
		else:
			output[i]+=1;
	return output;

dumpname = "algo_linint";

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


	f = open( "CV"+"algo_50k_8march"+".ms", "r" );
	CV1 = pickle.load(f);
	f.close();

	f = open( "CV"+"algo_600"+".ms", "r" );
	CV2 = pickle.load(f);
	f.close();

	load_learnt = open( "algo_50k_8march"+".ms", "r" );
	clf1 = pickle.load( load_learnt );
	load_learnt.close();

	load_learnt = open( "algo_600"+".ms", "r" );
	clf2 = pickle.load( load_learnt );
	load_learnt.close();
	count = 1;
	for row in targetcsv[:-600]:
		wordbag = splitter(row[1]);
		newf1 = float(clf1.predict( CV1.transform([ wordbag ]) )[0]);
		newf2 = float(clf2.predict( CV2.transform([ wordbag ]) )[0]);
		if(count <=  (len(targetcsv)-600)/2):
			datatest.append((row[0], [newf1, newf2]));
		else:
			data.append((row[0], [newf1, newf2]));
		count+=1;

	print "Done Spliting Data!";

	X = list(i[1] for i in data);
	y = np.array(list( i[0] for i in data));

	clf = linear_model.LogisticRegression();
	clf.fit(X, y);
	print "Learnt !... Ready to save in a file";
	if(dumpname != None):
		dump_learnt = open(dumpname+".ms", "wb");
		pickle.dump(clf, dump_learnt);
		dump_learnt.close();

	X1 = list(i[1] for i in datatest);
	y1 = clf.predict(X1);

	correct = sum((y1[i] == datatest[i][0]) for i in xrange(len(datatest)));
	print "correct = ", correct, "datatest = ", len(datatest);
	print "Accuracy = ", (correct*100.0)/len(datatest);
mainfunc();

