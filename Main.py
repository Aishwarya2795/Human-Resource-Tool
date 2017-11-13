# Importing the necessary libraries for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

#########################################

# Exploring the data - EDA
df = pd.read_csv('HR_comma_sep.csv', index_col=None)

df = df.rename(columns={'sales' : 'department'})

a=df.head()
print(df.isnull().any())
print(df.corr())
print(df.head())
print(df.describe())
turnover_rate = (df.left.value_counts() / len(df))*100
print("**************************************")
print(turnover_rate)
turnover_Summary = df.groupby('left')
print(turnover_Summary.mean())

# Correlation Matrix
corr = df.corr()
corr = (corr)
ax= plt.axes()
corr1=sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax)
ax.set_title('Correlation Matrix & Heatmap')
print(corr)

emp_population = df['satisfaction_level'][df['left'] == 0].mean()
emp_turnover_satisfaction = df[df['left']==1]['satisfaction_level'].mean()

print( 'The mean satisfaction for the employee population with no turnover is: ' + str(emp_population))
print( 'The mean satisfaction for employees that had a turnover is: ' + str(emp_turnover_satisfaction))

#sns.distplot(emp_population, kde=False, color="g",ax=axes[0]).set_title('mean satisfaction for the employee population with no turnover')

print("****************")

#############

f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(y="department", hue='left', data=df).set_title('Employee Department Turnover Distribution');

f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="salary", hue='left', data=df).set_title('Employee Salary Turnover Distribution');

##############

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(df.satisfaction_level, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')
axes[0].set_ylabel('Employee Count')

# Graph Employee Evaluation
sns.distplot(df.last_evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')
axes[1].set_ylabel('Employee Count')

# Graph Employee Average Monthly Hours
sns.distplot(df.average_montly_hours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
axes[2].set_ylabel('Employee Count')

print("********************")

###############################################

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3)
kmeans.fit(df[df.left==1][["satisfaction_level","last_evaluation"]])

colors = ['orange' if c == 0 else 'green' if c == 2 else 'red' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="satisfaction_level", y="last_evaluation", data=df[df.left==1],alpha=0.25,color=colors)
plt.xlabel("satisfaction_level")
plt.ylabel("last_evaluation")
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:,1], color="black", marker="X", s=100)
plt.title("Clusters of Employee Turnover")

###############################################

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

# Convert these variables into categorical variables
df["department"] = df["department"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes

# Create train and test splits
target_name = 'left'
X = df.drop('left', axis=1)


y=df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

dtree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train,y_train)
print(dtree)
## plot the importances ##
importances = dtree.feature_importances_
feat_names = df.drop(['left'],axis=1).columns


indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
#plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical', color='Green', fontsize=10)
plt.xlim([-1, len(indices)])
#plt.show()

dtree = tree.DecisionTreeClassifier(
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train,y_train)
print ("\n\n ---Decision Tree Model---")
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(y_test, dtree.predict(X_test)))

###########################################

plt.show()
