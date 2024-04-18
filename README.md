# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import dataset and get data info

2.Check for null values
3.Map values for position column

4.Split the dataset into train and test set

5.Import decision tree regressor and fit it for data

6.Calculate MSE value, R2 value and Data predict


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: POOJA.S
RegisterNumber:  212223040146
*/
import pandas as pd
data=pd.read_csv("/content/Salary_EX7.csv")
import matplotlib.pyplot as plt
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:
![Screenshot 2024-04-02 212519](https://github.com/Shubhavi17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150005085/71b64cba-5706-479e-8ee4-a1cb92e8f774)
![Screenshot 2024-04-02 212529](https://github.com/Shubhavi17/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150005085/7d44f57b-e253-44e6-ab5c-3479aae9da64)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
