import plotly.plotly as py
from plotly.graph_objs import *
from HumanResource import mlp,adaboost,kneighbours
from plotly.offline import plot
import pandas as pd
import numpy.ma


data2 = pd.read_csv("/Users/aishwaryagunashekar/Desktop/HR-project/Hr-master/HR_comma_sep.csv")
data2 = data2.rename(columns={'sales': 'department'})
data2.dtypes

data2['salary'] = data2['salary'].astype('str')

new_df = data2

c = len(new_df.index) + 1

#List unique values in the df['name'] column
new_df.salary.unique()
new_df.department.unique()

#new_df['salary'] = new_df['salary'].map({'low': 0, 'medium': 1, 'high': 2})
#new_df['department'] = new_df['department'].map({'sales': 1, 'accounting': 2, 'hr': 3, 'technical':
# 4,'support': 5,'management': 6,'IT': 7,'product_mg': 8,'marketing': 9,'RandD': 10 })

salary_dept = data2.filter(['salary','department','satisfaction_level'], axis=1)

lowSal = salary_dept[(salary_dept['salary'] =='low')]
lowDept = lowSal.groupby('department').mean()
satList1 = list(lowDept['satisfaction_level'])

medSal = salary_dept[(salary_dept['salary']=='medium')]
medDept = medSal.groupby('department').mean()
satList2 = list(medDept['satisfaction_level'])

highSal = salary_dept[(salary_dept['salary']=='high')]
highDept = highSal.groupby('department').mean()
satList3 = list(highDept['satisfaction_level

salaryList = []
departmentList =[]
salaryList = new_df['salary'].tolist()
departmentList = new_df['department'].tolist()

salaryListLeft = []
salaryListLeft = new_df[(new_df['left']==1)]['salary']

salaryListNotLeft = []
salaryListNotLeft = new_df[(new_df['left']==0)]['salary']

#plot 5 ( two sub plots )

sal_satis = data2.filter(['salary','satisfaction_level'], axis=1)
dep_satis = data2.filter(['department','satisfaction_level'], axis=1)

satis_list = []
satis_list = satList1 + satList2 +satList3
labels3 = satis_list

grouped1 = sal_satis.groupby('satisfaction_level').count()
grouped2 = dep_satis.groupby('satisfaction_level').count()


values4a = []
values4b = []

for satis in grouped1.index:
    values4a = values4a + list(grouped1.loc[satis])

for satis in grouped2.index:
    values4b = values4b + list(grouped2.loc[satis])



trace1 = {"x": salaryListLeft,
          "y": satis_list,
          "marker": {"color": "pink", "size": 12},
          "mode": "markers",
          "name": "Left",
          "type": "scatter"
}

trace2 = {"x": salaryListNotLeft,
          "y": satis_list,
          "marker": {"color": "blue", "size": 12},
          "mode": "markers",
          "name": "Men", 
          "type": "scatter",
}

data = Data([trace1, trace2])
layout = {"title": "Gender Earnings Disparity",
          "xaxis": {"title": "Annual Salary (in thousands)", },
          "yaxis": {"title": "School"}}

fig = Figure(data=data, layout=layout)
py.iplot(fig, filenmae='basic_dot-plot')
