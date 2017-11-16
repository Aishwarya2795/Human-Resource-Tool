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
"""
for x in range(1,c):
    new_df[new_df.salary=="low"].salary = 0
    new_df[new_df.salary=="medium"].salary = 1
    new_df[new_df.salary=="high"].salary = 2

new_df.department[new_df.department=="sales"] = 1
    new_df[new_df.department=="accounting",department] = 2
    new_df[new_df.department=="hr"].department = 3
    new_df[new_df.department=="technical"].department = 4
    new_df[new_df.department=="support"].department = 5
    new_df[new_df.department=="management"].department = 6
    new_df[new_df.department=="IT"].department = 7
    new_df[new_df.department=="product_mg"].department = 8
    new_df[new_df.department=="marketing"].department = 9
    new_df[new_df.department=="RandD"].department = 10
"""

#new_df.salary.replace(to_replace=dict(low=0, medium=1, high=2), inplace=True)

new_df['salary'] = new_df['salary'].map({'low': 0, 'medium': 1, 'high': 2})
new_df['department'] = new_df['department'].map({'sales': 1, 'accounting': 2, 'hr': 3, 'technical':
 4,'support': 5,'management': 6,'IT': 7,'product_mg': 8,'marketing': 9,'RandD': 10 })

salary_dept = data2.filter(['salary','department','satisfaction_level'], axis=1)

lowSal = salary_dept[(salary_dept['salary']=='low')]
lowDept = lowSal.groupby('department').mean()
satList1 = list(lowDept['satisfaction_level'])

medSal = salary_dept[(salary_dept['salary']=='medium')]
medDept = medSal.groupby('department').mean()
satList2 = list(medDept['satisfaction_level'])

highSal = salary_dept[(salary_dept['salary']=='high')]
highDept = highSal.groupby('department').mean()
satList3 = list(highDept['satisfaction_level'])



#new_df.salary.replace(['low', 'medium','high'], [0, 1, 2], inplace=True)

salaryList = []
departmentList =[]
salaryList = new_df['salary'].tolist()
departmentList = new_df['department'].tolist()

deptUniqueList = []
deptUniqueList = list(lowDept.department.unique())

#plot1 - Heatmap of Salary and Department against Satisfaction Level of Employees

SalaryVSDepartment = Heatmap(
    # z should have number of elements equal to y, inner list should have number of elements in x
    z = [satList1,satList2,satList3],
    y= ['low','medium','high'],
    x= ['IT','RandD','accounting','hr','management','marketing','product_mg','sales','support','technical']
)

layout= Layout(
    title= 'Salary and Department against Satisfaction Level (Heat Gradient) of Employees',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Departments',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Salary Level',
        ticklen= 5,
        gridwidth= 2,
    ),

    showlegend= False
)
data1 = [SalaryVSDepartment]
fig_comp = Figure( data=data1, layout=layout)

#data1 = Data([trace0, trace1])

#For online plotting
py.plot(fig_comp, filename = 'salary-department-satisfaction-online')

#For offline plotting
#plot(data1, filename = 'salary-department-satisfaction-line')

#plot number 2 - Histogram

mlptrace = Histogram(
	y=mlp,
    histnorm='count',
    name='MLP Classifier',
    opacity=0.75
)
adaboosttrace = Histogram(
	y=adaboost,
    histnorm='count',
    name='AdaBoost Classifier',
    opacity=0.75
)
kntrace = Histogram(
	y=kneighbours,
    histnorm='count',
    name='KNeighbours Classifier',
    opacity=0.75
)

layout2 = Layout(
    title='Prediction Rate Results',
    yaxis=dict(
        title='Classifiers'
    ),
    bargap=0.2,
    bargroupgap=0.1
)

)

data3 = [mlptrace,adaboosttrace,kntrace]
fig_comp2 = Figure( data=data3, layout = layout2)

py.plot(fig_comp2, filename='basic histogram')


#plot3 - Histogram Salary Distribution

labels1 = ['low','medium','high']
salary_dept.groupby('salary').count()
values1 = [7316,6446,1237]

trace1 = Pie(labels=labels1, values=values1)

#fig_comp3 = Figure( data=trace1, layout = layout3)

py.plot([trace1], filename='salary_pie_chart')

#plot4 - Department strength distribution

labels2 = ['IT','RandD','accounting','hr','management','marketing','product_mg','sales','support','technical']
salary_dept.groupby('department').count()

values2 = [1227,787,767,739,630,858,902,4140,2229,2720]

trace2 = Pie(labels=labels2, values=values2)

#fig_comp3 = Figure( data=trace1, layout = layout3)

py.plot([trace2], filename='dept_pie_chart')

#plot5
satis_list = []
satis_list = satList1 + satList2 +satList3
labels3 = satis_list
grouped = salary_dept.groupby('satisfaction_level').count()

values3 = [1227,787,767,739,630,858,902,4140,2229,2720]

trace3 = Pie(labels=labels3, values=values3)

#fig_comp3 = Figure( data=trace1, layout = layout3)

py.plot([trace2], filename='dept_pie_chart')

