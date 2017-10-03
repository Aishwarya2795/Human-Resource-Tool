import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.neural_network
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import numpy.ma
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import column_or_1d
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv("D:\Coursework\SSDI\Project\Dataset\HR_comma_sep.csv")
#print ("********************")
columns = data[0:0]
#df = DataFrame(data = data, c)
testdata = data[14001:-1]
trainData = data[0:14000]

#print (testdata)
#features = trainData.columns[:-1]
x=trainData.iloc[:,-1]
y=trainData.iloc[:,0:-2]

#clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), random_state=1,activation='relu',batch_size ='auto',learning_rate='constant',verbose=False,warm_start =True,max_iter=1000,learning_rate_init =0.005,shuffle=True )
#clf = tree.DecisionTreeClassifier()
#clf = NearestNeighbors(n_neighbors=5, algorithm='auto')
clf =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=6, learning_rate=0.01,algorithm="SAMME")
#scores = cross_val_score(clf, y, x, cv=5)
clf.fit(y,x)
#print ("Cross validation with 5 folds ")
#print (scores)


testResult=testdata.iloc[:,-1]
testfeatures = list(testdata[:-1])
testFeature=testdata.iloc[:,0:-2]
actualResult=list(testResult)
#print(actualResult)
#output = clf.predict_proba(testFeature)
#print output
#outFile = open('Output_11_10_1_09.txt','w')
#outFile.write(output)
#outFile.close()
predictedData=list(clf.predict(testFeature))
s=f=0
for i in range(len(actualResult)-2):
	if actualResult[i] == predictedData[i]:
		s=s+1
	else:
		f=f+1
print ("Prediction rate is "+str(float(s)/len(actualResult)*100.0))
