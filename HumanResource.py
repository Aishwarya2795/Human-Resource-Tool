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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def prediction(actualResult, predictedData, algorithm):
	s=f=0
	for i in range(len(actualResult)-2):
		if actualResult[i] == predictedData[i]:
			s=s+1
		else:
			f=f+1
	print ("Prediction rate for "+str(algorithm) +" is:"+str(float(s)/len(actualResult)*100.0))


df = pd.read_csv("D:\Coursework\SSDI\Project\Dataset\HR_comma_sep.csv")
df.rename(columns={'sales': 'department'}, inplace=True)  # Rename 'sales' column as 'department'

# column names Saved
column_names = df.columns.values
column_names = column_names[column_names!='left']  # Remove the 'left' column
'''
index = np.argwhere(x==3)
y = np.delete(x, index)	
'''
# pickle.dump(column_names, open('column_names.p', 'wb'))
np.save('column_names.npy', column_names)


le = LabelEncoder()
train = df['department'].unique().tolist()
test = df['department']
df['department'] = le.fit(train).transform(test)
# salary column...
train = df['salary'].unique().tolist()
test = df['salary']
df['salary'] = le.fit(train).transform(test)

#print (df)
df = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident', 'promotion_last_5years','department','salary','left']]

#df[['number_project','average_montly_hours', 'time_spend_company', 'department', 'salary']] = df[['number_project','average_montly_hours', 'time_spend_company', 'department', 'salary']].apply(lambda x: StandardScaler().fit_transform(x))


scaled_features = df.copy()
col_names = ['number_project','average_montly_hours', 'time_spend_company', 'department', 'salary']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
'''
print (scaled_features.values)
print ("Dataframe")
print (df.values)

	
data = pd.read_csv("D:\Coursework\SSDI\Project\Dataset\HR_comma_sep.csv")
#print ("********************")
columns = data[0:0]
#df = DataFrame(data = data, c)
testdata = data[14001:-1]
trainData = data[0:14000]
'''
num_columns = len(scaled_features.columns)
array = scaled_features.values
X = array[:, 0:(num_columns - 1)].astype(float)

Y = array[:, (num_columns - 1)]


test_size = 0.33
seed = 7
x, testFeature, y , testResult = train_test_split(X, Y, test_size=test_size, random_state=seed)

'''
trainData = df
#print (testdata)
#features = trainData.columns[:-1]
lable=trainData.iloc[:,-1]
trainFeature=trainData.iloc[:,0:-2]
x, testFeature, yy , testResult = train_test_split(trainFeature, lable, test_size=0.33, random_state=7)
#testResult=testResult.iloc[:,-1]
#testFeature=testdata.iloc[:,0:-2]
''' 

actualResult=list(testResult)

clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), random_state=1,activation='relu',batch_size ='auto',learning_rate='constant',verbose=False,warm_start =True,max_iter=1000,learning_rate_init =0.01,shuffle=True )
clf.fit(x,y)
predictedData=list(clf.predict(testFeature))
prediction(actualResult, predictedData, "Neural Network Algorithm")
#clf = tree.DecisionTreeClassifier()

clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
clf.fit(x,y)
predictedData=list(clf.predict(testFeature))
prediction(actualResult, predictedData, "KNearestNeighbors Algorithm")


clf =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=6, learning_rate=0.01,algorithm="SAMME")
clf.fit(x,y)
predictedData=list(clf.predict(testFeature))
prediction(actualResult, predictedData, "AdaBoostClassifier")
#scores = cross_val_score(clf, y, x, cv=5)
#print ("Cross validation with 5 folds ")
#print (scores)



'''
s=f=0
for i in range(len(actualResult)-2):
	if actualResult[i] == predictedData[i]:
		s=s+1
	else:
		f=f+1
print ("Prediction rate is "+str(float(s)/len(actualResult)*100.0))
'''
