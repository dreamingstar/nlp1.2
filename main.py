import numpy as np
import sklearn
from sklearn.feature_extraction import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import linear_model
from nltk.tokenize import TweetTokenizer
from nltk.sentiment import *
import csv,re,random, math, nltk, pickle

random.seed(1296441);
tknzr = TweetTokenizer();

def clean_string(inp):
	return "".join(i if 0 <= ord(i) < 128 else " " for i in inp);

def splitter(inp):
	inp = re.sub("(https?://[^ ]*)|([@][a-zA-Z0-9_]+)|(&[a-zA-Z0-9]+;)", "", inp.lower()); #Remove URL, @<Username>, &nbsp;
	tokens = tknzr.tokenize(clean_string(inp));
	tokens = nltk.sentiment.util.mark_negation(tokens)
	return tokens;

_isloading = False;
dumpname = "algo_target600";

def read_csv(fn, isdoshuufle = False):
	csvfd = open(fn, 'rb');
	sp = csv.reader(csvfd)
	csvdata =  list(row for row in sp);
	csvfd.close();
	if(isdoshuufle):
		random.shuffle(csvdata);
	return csvdata;


def mainfunc():
	data = [];
	datatest = [];


#	sourcecsv = read_csv('../training.csv', True);
	targetcsv = read_csv('modi_train.csv', True);


#	csvdata = csvdata[:50000]

	print "Done the shuffle";

	if(False):
		count = 1;
		for row in csvdata:
			wordbag = splitter(row[1]);
			if(count <= (len(csvdata)*10)/100):
				datatest.append((row[0], row[1], wordbag))
			else:
				if(_isloading):
					break;
				data.append((row[0], row[1], wordbag));
			if(count % 50000 == 0):
				print count;
			count += 1;
	elif(False): #Source only modal
		for row in targetcsv:
			wordbag = splitter(row[1]);
			datatest.append((row[0], row[1], wordbag));
	elif(True):
		count = 1;
		for row in targetcsv:
			wordbag = splitter(row[1]);
			if(count <=  (len(targetcsv)*12)/100 ):
				datatest.append((row[0], row[1], wordbag));
			else:
				data.append((row[0], row[1], wordbag));
			count+=1;

	print "Done Spliting Data!";

	if(not(_isloading)):
		CV = sklearn.feature_extraction.text.CountVectorizer()
		X = CV.fit_transform(list(" ".join(i[2]) for i in data))
		y = np.array(list(i[0] for i in data));
		f = open( "CV"+dumpname+".ms", "wb" );
		pickle.dump(CV, f);
		f.close();
	else:
		f = open( "CV"+dumpname+".ms", "r" );
		CV = pickle.load(f);
		f.close();


	if(not(_isloading)):
		if(False):
			clf = MultinomialNB();
		elif(False):
			clf = linear_model.LogisticRegression();
		elif(True):
			clf = LinearSVC();
		clf.fit(X, y);
		print "Learnt !... Ready to save in a file";
		dump_learnt = open(dumpname+".ms", "wb");
		pickle.dump(clf, dump_learnt);
		dump_learnt.close();
	else:
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

