#!/usr/bin/env python
 
import csv

def numline(file):
	with open('input.csv', 'r') as csvfile:
		csvData = csv.reader(csvfile, delimiter=',')
		csvData.next()
	 	numlines=0
		for row in csvData:
			numlines = numlines+1
			
	return numlines


lines = 30
with open('input.csv', 'r') as csvfile:
	csvData = csv.reader(csvfile, delimiter=',')
	csvData.next()
	numlines=numline('input.csv');
	test = numlines*lines/100
	train  = numlines-test
	tef = open('testgiven.csv', 'wb') 
	trf = open('traingiven.csv', 'wb')
	writer_tef= csv.writer(tef)
	writer_trf = csv.writer(trf)
	for row in csvData:
		if test >0:
			writer_tef.writerow(row)
			test = test-1
		else:
			writer_trf.writerow(row)

