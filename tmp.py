import csv, random


csvfd = open('../training.csv', 'rb');
sp = csv.reader(csvfd)
csvdata =  list(row for row in sp);
csvfd.close();


random.shuffle(csvdata);

f_inp = open("input1.txt", "w");
f_outp = open("outp_real1.txt", "w");

for i in xrange(100000):
	f_inp.write(csvdata[i][1]+"\n");
	f_outp.write(csvdata[i][0]+"\n");

f_inp.close();
f_outp.close();
