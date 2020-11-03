import utils
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import math	

train=pd.read_csv("train.csv")

def getPrior(data):
	som = 0
	for e in data["target"]:
		if e == 1:
			som += 1
	p = som/len(data.target)
	inf = p-1.96*math.sqrt(p*(1-p)/len(data.target))
	sup = p+1.96*math.sqrt(p*(1-p)/len(data.target))
	return {"estimation" : p, "min5pourcent" : inf, "max5pourcent" : sup}


class APrioriClassifier(utils.AbstractClassifier):

	def __init__(self):
		pass

	def estimClass(self, attrs):
		return 1

	def statsOnDf(self, data):
		estim = self.estimClass(None)
		vp = 0
		vn = 0
		fp = 0
		fn = 0
		for e in data["target"]:
			if e == 0:
				fp+=1
			else:
				vp+=1
		return {"VP" : vp, "VN" : vn, "FP" : fp, "FN" : fn, "Précision" : vp/(vp+fp), "Rappel" : vp/(vp+fn)}


def P2D_l(df, attr):
	l = pd.crosstab(df.target, [df[attr]])
	sm0 = 0
	sm1 = 0
	for i in range(len(l)):
		sm0 += l[i]
		sm1 += l[i]
	print(sm0)

print(P2D_l(train, "thal"))