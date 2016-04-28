import csv
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import linear_model
from numpy import nan
from sklearn import svm
from itertools import izip

csize=10

priceopen=[]
priceclose=[]
pricelow=[]
pricehigh=[]
volumeopen=[]
volumeclose=[]
volumelow=[]
volumehigh=[]
startdate=[]
enddate=[]

with open('traingiven.csv', 'r') as csvfile:
	csvData = csv.reader(csvfile, delimiter=',')
	csvData.next()
	i=0
	temppricelow=2147483647
	temppricehigh=0
	tempvolumelow=2147483647
	tempvolumehigh=0
	arraysize=0
	for row in csvData:
		if i==0:
			priceclose.append(float(row[4]))
			volumeclose.append(int(row[5]))
			enddate.append(str(row[0]))

		temppricelow=min(temppricelow,float(row[3]))
		temppricehigh=max(temppricehigh,float(row[2]))
		tempvolumelow=min(tempvolumelow,int(row[5]))
		tempvolumehigh=max(tempvolumehigh,int(row[5]))

		i=i+1

		if i==csize:
			pricehigh.append(float(temppricehigh))
			temppricehigh=0
			pricelow.append(float(temppricelow))
			temppricelow=2147483647
			priceopen.append(float(row[1]))
			volumehigh.append(int(tempvolumehigh))
			tempvolumehigh=0
			volumelow.append(int(tempvolumelow))
			tempvolumelow=2147483647
			volumeopen.append(int(row[5]))
			startdate.append(str(row[0]))
			arraysize=arraysize+1
			i=0

with open('trainactual.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Start date','End date','Open Price','Close Price','Low Price','High Price','Open Volume','Close Volume','Low Volume','High Volume',])
    for i1 in range(arraysize):
    	i=arraysize-1-i1
    	writer.writerow((startdate[i],enddate[i],priceopen[i],priceclose[i],pricelow[i],pricehigh[i],volumeopen[i],volumeclose[i],volumelow[i],volumehigh[i]))

