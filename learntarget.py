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


#random.seed(266);

random.seed(982382386);


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


ht_replace = dict(i.lower().split(":") for i in  """#arunjaitley:AJ
#ArunJaitley:AJ
#ArvindKejriwal:AK
#BJD:MODI
#bjp:MODI
#crazyWall:AK
#Kejri:AK
#Kejriwal:AK
#Khujliwala:AK
#Modiji:modi
#MoronendraModi:modi
#NakaraModi:modi
#namo:modi
#Namo:modi
#NaMo:modi
#NaMO:modi
#NAMO:modi
#NaMoAnd:modi
#NaMoClick:modi
#narendramodi:modi
#Narendramodi:modi
#NarendraModi:modi
#NarendraModiMore:modi
#PM:modi
#PMO:modi
#PMofIndia:modi
#PMOIndia:modi
#primeminister:modi
#PrimeMinister:modi
#government:modi
#Government:modi
#Hinduism:hindu
#hindus:hindu
#hindutva:hindu
#Hindutva:hindu
#hinduvadi:hindu
#hinduyouthsummit2016:hindu
#BJPBengal:MODI
#bjpgovt:MODI
#BJPhttps:MODI
#BJPigs:MODI""".split("\n"));

good_words = """#Betheforce
#Help
#HelpInIndia
#GoodGovernance
#FrustratedAAP
#committed
#CleanDelhi
#cleanindia
#CleanIndia
#CheerLeaders
#GetWellSoon
#GoodMorning
#Intolerance
#lolbjp
#SkillIndia
#Salute
#Respect
#lolmodi
#Losar
#loss
#ModiBhakts
#ModiFuelLoot
#ModiMurdabad
#Nauntaki
#Scams
#HarHarModi
#BetiBachaoBetiPadhao
#bannerjilovesUNaMo
#acchedin
#AccheDin
#achedin
#AchheDin
#ProudIndian
#proudofyoumodi
#progress
#Progress
#NationLovesModi
#NamoNeed
#NaMoInspiresMe
#Motivation
#ModiWithYouth
#MODIfied
#ManKiBaat
#makeindiaproud
#MakeinIndia
#MakeInIndia
#lifechanging
#Jai
#JaiHind
#JaiJagannath
#InspirationalVideo
#indiawithnamo
#IndiaShining
s#IndiaIsTolerant
#indiafirst
#increableindia
#IncredibleIndia
#Aarav
#AbKiBaarModiSarkar
#AAPians
#AAPtard
#AAPtards
#1YearOfHonestPolitics""".lower().split("\n");

bad_words = """#1YearOfRayataPolitics
#2ndPappu
#AndhBhakts
#Bhakats
#Bhaktards
#Bhaktology
#Bhakts
#BHAKTS
#BJPLandGrab
#BrainDeadBhakts
#FakeInIndia
#feku
#Feku
#FekuBhakts
#FekuModi
#HateinIndia""".lower().split("\n");




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
	data = [];
	datatest = [];

	targetcsv = read_csv('modi_train.csv', True);

	f = open("CV"+"algo_50k_8march"+".ms", "r");
	CV = pickle.load(f);
	f.close();

	load_learnt = open( "algo_50k_8march"+".ms", "r" );
	clf = pickle.load( load_learnt );
	load_learnt.close();

	print "Done the shuffle";
	count = 1;
	for row in targetcsv:
		wordbag = splitter(row[1]);
#		wordbag["sourcef"] = clf.predict(CV.transform([wordbag]))
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
#	y1 = clf.predict(X1);
	y1 = predict_modi(clf, X1, list(i[2] for i in datatest));

	correct = sum((y1[i] == datatest[i][0]) for i in xrange(len(datatest)));
	print "correct = ", correct, "datatest = ", len(datatest);
	print "Accuracy = ", (correct*100.0)/len(datatest);
mainfunc();

