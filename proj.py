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

lambda1=5

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

priceopen2=[]
priceclose2=[]
pricelow2=[]
pricehigh2=[]
volumeopen2=[]
volumeclose2=[]
volumelow2=[]
volumehigh2=[]
startdate2=[]
enddate2=[]

arraysize=0
arraysize2=0

with open('trainactual.csv', 'r') as csvfile:
	csvData = csv.reader(csvfile, delimiter=',')
	csvData.next()
	for row in csvData:
		priceopen.append(float(row[2]))
		priceclose.append(float(row[3]))
		pricelow.append(float(row[4]))
		pricehigh.append(float(row[5]))
		volumeopen.append(int(row[6]))
		volumeclose.append(int(row[7]))
		volumelow.append(int(row[8]))
		volumehigh.append(int(row[9]))
		arraysize=arraysize+1

with open('testactual.csv', 'r') as csvfile:
	csvData = csv.reader(csvfile, delimiter=',')
	csvData.next()
	for row in csvData:
		priceopen2.append(float(row[2]))
		priceclose2.append(float(row[3]))
		pricelow2.append(float(row[4]))
		pricehigh2.append(float(row[5]))
		volumeopen2.append(int(row[6]))
		volumeclose2.append(int(row[7]))
		volumelow2.append(int(row[8]))
		volumehigh2.append(int(row[9]))
		arraysize2=arraysize2+1

feature1=[]
feature2=[]
feature3=[]
feature4=[]
feature5=[]
feature6=[]
pricehighlambda=[]
pricelowlambda=[]
volumehighlambda=[]
volumelowlambda=[]
feature11=[]
feature12=[]

tsfeature1=[]
tsfeature2=[]
tsfeature3=[]
tsfeature4=[]
tsfeature5=[]
tsfeature6=[]
tspricehighlambda2=[]
tspricelowlambda2=[]
tsvolumehighlambda2=[]
tsvolumelowlambda2=[]
tsfeature11=[]
tsfeature12=[]

yval=[]
yvalactual=[]

i=0
for i in range(arraysize):
	if i==0:
		feature1.append(float(0))
		feature4.append(float(0))
	else:
		feature1.append(float(priceopen[i]-priceopen[i-1])/float(priceopen[i-1]))
		feature4.append(float(volumeopen[i]-volumeopen[i-1])/float(volumeopen[i-1]))

	if i<2:
		feature2.append(float(0))
		feature3.append(float(0))
		feature5.append(float(0))
		feature6.append(float(0))		
	else:
		feature2.append(float(pricehigh[i-1]-pricehigh[i-2])/float(pricehigh[i-2]))
		feature3.append(float(pricelow[i-1]-pricelow[i-2])/float(pricelow[i-2]))
		feature5.append(float(volumehigh[i-1]-volumehigh[i-2])/float(volumehigh[i-2]))
		feature6.append(float(volumelow[i-1]-volumelow[i-2])/float(volumelow[i-2]))

	temppricehighlambda=float(0)
	temppricelowlambda=float(2147483647)
	tempvolumehighlambda=float(0)
	tempvolumelowlambda=float(2147483647)
	for i1 in xrange(i-lambda1,i):
		if i1>=0:
			temppricehighlambda=max(temppricehighlambda,float(pricehigh[i1]))
			temppricelowlambda=min(temppricelowlambda,float(pricelow[i1]))
			tempvolumehighlambda=max(tempvolumehighlambda,float(volumehigh[i1]))
			tempvolumelowlambda=min(tempvolumelowlambda,float(volumelow[i1]))

	pricehighlambda.append(float(temppricehighlambda))
	pricelowlambda.append(float(temppricelowlambda))
	volumehighlambda.append(float(tempvolumehighlambda))
	volumelowlambda.append(float(tempvolumelowlambda))
	#print priceopen[i],',',priceopen[i-1],',',temppricehighlambda,',',temppricehighlambda,',',float(priceopen[i]-priceopen[i-1])/float(temppricehighlambda-temppricelowlambda)
	if i==0:
		feature11.append(float(0))
		feature12.append(float(0))
	else:
		if temppricehighlambda==temppricelowlambda:
			feature11.append(float(0))
		else:
			feature11.append(float(priceopen[i]-priceopen[i-1])/float(temppricehighlambda-temppricelowlambda))
		if tempvolumehighlambda==tempvolumelowlambda:
			feature12.append(float(0))
		else:
			feature12.append(float(volumeopen[i]-volumeopen[i-1])/float(tempvolumehighlambda-tempvolumelowlambda))
	if priceclose[i]>priceopen[i]:
		yval.append(int(1))
	else:
		yval.append(int(0))

i2=0
for i2 in range(arraysize2):
	if i2==0:
		tsfeature1.append(float(0))
		tsfeature4.append(float(0))
	else:
		tsfeature1.append(float(priceopen2[i2]-priceopen2[i2-1])/float(priceopen2[i2-1]))
		tsfeature4.append(float(volumeopen2[i2]-volumeopen2[i2-1])/float(volumeopen2[i2-1]))

	if i2<2:
		tsfeature2.append(float(0))
		tsfeature3.append(float(0))
		tsfeature5.append(float(0))
		tsfeature6.append(float(0))		
	else:
		tsfeature2.append(float(pricehigh2[i2-1]-pricehigh2[i2-2])/float(pricehigh2[i2-2]))
		tsfeature3.append(float(pricelow2[i2-1]-pricelow2[i2-2])/float(pricelow2[i2-2]))
		tsfeature5.append(float(volumehigh2[i2-1]-volumehigh2[i2-2])/float(volumehigh2[i2-2]))
		tsfeature6.append(float(volumelow2[i2-1]-volumelow2[i2-2])/float(volumelow2[i2-2]))

	temppricehighlambda=float(0)
	temppricelowlambda=float(2147483647)
	tempvolumehighlambda=float(0)
	tempvolumelowlambda=float(2147483647)
	for i3 in xrange(i2-lambda1,i2):
		if i3>=0:
			temppricehighlambda=max(temppricehighlambda,float(pricehigh[i3]))
			temppricelowlambda=min(temppricelowlambda,float(pricelow[i3]))
			tempvolumehighlambda=max(tempvolumehighlambda,float(volumehigh[i3]))
			tempvolumelowlambda=min(tempvolumelowlambda,float(volumelow[i3]))

	tspricehighlambda2.append(float(temppricehighlambda))
	tspricelowlambda2.append(float(temppricelowlambda))
	tsvolumehighlambda2.append(float(tempvolumehighlambda))
	tsvolumelowlambda2.append(float(tempvolumelowlambda))
	
	if i2==0:
		tsfeature11.append(float(0))
		tsfeature12.append(float(0))
	else:
		if temppricehighlambda==temppricelowlambda:
			tsfeature11.append(float(0))
		else:
			tsfeature11.append(float(priceopen2[i2]-priceopen2[i2-1])/float(temppricehighlambda-temppricelowlambda))
		if tempvolumehighlambda==tempvolumelowlambda:
			tsfeature12.append(float(0))
		else:
			tsfeature12.append(float(volumeopen2[i2]-volumeopen2[i2-1])/float(tempvolumehighlambda-tempvolumelowlambda))
	if priceclose2[i2]>priceopen2[i2]:
		yvalactual.append(int(1))
	else:
		yvalactual.append(int(0))

X = np.vstack((feature1,feature2,feature3,feature4,feature5,feature6,feature11,feature12)).T
X2 = np.vstack((tsfeature1,tsfeature2,tsfeature3,tsfeature4,tsfeature5,tsfeature6,tsfeature11,tsfeature12)).T

clf=svm.SVC()
clf.fit(X,yval)

yvalpredicted = clf.predict(X2)

resultFile=open("output.csv",'wb')
wr=csv.writer(resultFile, dialect='excel')
wr.writerow(['Id','y-predicted','y-actual',])
id1=0
for i in range(len(yvalpredicted)):
	wr.writerow([id1,int(yvalpredicted[i]),int(yvalactual[i])])
	id1=id1+1

tn=0
fn=0
fp=0
tp=0
for i in range(len(yvalactual)):
	if yvalactual[i]==0 and yvalpredicted[i]==0:
		tn=tn+1
	elif yvalactual[i]==1 and yvalpredicted[i]==0:
		fn=fn+1
	elif yvalactual[i]==0 and yvalpredicted[i]==1:
		fp=fp+1
	elif yvalactual[i]==1 and yvalpredicted[i]==1:
		tp=tp+1

accuracy=float(tn+tp)/float(tn+tp+fn+fp)
precision=float(tn+tp)/float(tn+tp+fn+fp)
recall=float(tp)/float(tp+fp)

profit=0
for i in range(len(yvalpredicted)):
	if yvalpredicted[i]==1:
		profit=profit+(priceclose2[i]-priceopen2[i])

print profit,' ',accuracy, ' ', precision, ' ', recall


