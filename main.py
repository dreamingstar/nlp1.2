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

_isloading = True;
dumpname = "algo_15l";

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
	targetcsv = read_csv('modi_train.csv', True);

	sourcecsv = sourcecsv[:50000]

	print "Done the shuffle";
	if(False):
		count = 1;
		for row in sourcecsv:
			wordbag = splitter(row[1]);
			if(count <= (len(sourcecsv)*10)/100):
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
	elif(False): #Linear Weight
		count = 1;
		tdata = [];
		for row in targetcsv:
			wordbag = splitter(row[1]);
			if(count <= (len(targetcsv)*10)/100):
				datatest.append((row[0], row[1], wordbag))
			else:
				tdata.append((row[0], row[1], wordbag));
			count += 1;
		data+=tdata;
		weight += [len(sourcecsv)/len(tdata)]*len(tdata);
		for row in sourcecsv:
			wordbag = splitter(row[1]);
			data.append((row[0], row[1], wordbag));
			weight.append(1);
			if(count % 50000 == 0):
				print count;
			count += 1;
	elif(False): #Target Only
		count = 1;
		for row in targetcsv:
			wordbag = splitter(row[1]);
			if(count <=  120 ):
				datatest.append((row[0], row[1], wordbag));
			else:
				data.append((row[0], row[1], wordbag));
			count+=1;
	elif(False):
		f = open( "CV"+dumpname+".ms", "r" );
		CV = pickle.load(f);
		f.close();

		load_learnt = open( dumpname+".ms", "r" );
		clf = pickle.load( load_learnt );
		load_learnt.close();

		print clf.predict_proba(CV.transform([" ".join(splitter("mohit"))]));


		print clf.get_params();
		return;
		# X1 = CV.transform(list(" ".join(i[2]) for i in datatest))
		# y1 = clf.predict(X1);
		count = 1;
		for row in targetcsv:
			wordbag = splitter(row[1]);
			newf = "SpecialFeature"+clf.predict( CV.transform([ " ".join(wordbag)  ]) )[0];
			wordbag += [newf];
			if(count <=  (len(targetcsv)*12)/100 ):
				datatest.append((row[0], row[1], wordbag));
			else:
				data.append((row[0], row[1], wordbag));
			count+=1;

	elif(True): #LinInt
		f = open( "CV"+"algo_15l"+".ms", "r" );
		CV1 = pickle.load(f);
		f.close();

		f = open( "CV"+"algo_target600"+".ms", "r" );
		CV2 = pickle.load(f);
		f.close();

		load_learnt = open( "algo_15l"+".ms", "r" );
		clf1 = pickle.load( load_learnt );
		load_learnt.close();

		load_learnt = open( "algo_target600"+".ms", "r" );
		clf2 = pickle.load( load_learnt );
		load_learnt.close();
		count = 1;
		for row in targetcsv[:121]:
			wordbag = splitter(row[1]);
			newf1 = float(clf1.predict( CV1.transform([ " ".join(wordbag)  ]) )[0]);
			newf2 = float(clf2.predict( CV2.transform([ " ".join(wordbag)  ]) )[0]);
			if(count <=  40):
				datatest.append((row[0], [newf1, newf2]));
			else:
				data.append((row[0], [newf1, newf2]));
			count+=1;

	print "Done Spliting Data!";

	if(not(_isloading)):
		if(False):
			CV = sklearn.feature_extraction.text.CountVectorizer()
			X = CV.fit_transform(list(" ".join(i[2]) for i in data))
			y = np.array(list( i[0] for i in data));
		else:
			X = list(np.array(i[1]) for i in data);
			y = list(i[0] for i in data);
		if(False):
			f = open("CV"+dumpname+".ms", "wb");
			pickle.dump(CV, f);
			f.close();
	else:
		f = open("CV"+dumpname+".ms", "r");
		CV = pickle.load(f);
		f.close();


	if(not(_isloading)):
		if(False):
			clf = MultinomialNB();
		elif(True):
			clf = linear_model.LogisticRegression();
		elif(False):
			clf = LinearSVC();
		clf.fit(X, y);
		print "Learnt !... Ready to save in a file";
		if(False):
			dump_learnt = open(dumpname+".ms", "wb");
			pickle.dump(clf, dump_learnt);
			dump_learnt.close();
	else:
		load_learnt = open( dumpname+".ms", "r" );
		clf = pickle.load( load_learnt );
		load_learnt.close();


	if(False):
		X1 = CV.transform(list(" ".join(i[2]) for i in datatest))
		y1 = clf.predict(X1);
	else:
		X1 = np.array(list(np.array(i[1]) for i in datatest));
		y1 = clf.predict(X1);


	correct = sum((y1[i] == datatest[i][0]) for i in xrange(len(datatest)));
	print "correct = ", correct, "datatest = ", len(datatest);
	if(True):
		print "Accuracy = ", (correct*100.0)/len(datatest);
	else:
		print "Accuracy = ", (correct*100.0)/sum((i[0] in ["0", "4"]) for i in datatest);


mainfunc();

