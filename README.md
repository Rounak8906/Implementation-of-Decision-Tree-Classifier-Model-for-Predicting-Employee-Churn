# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ROUNAK SINGH
RegisterNumber:  212225240125
*/
```
```py
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```

```py
print("data.info():")
data.info()
```

```py
print("isnull() and sum():")
data.isnull().sum()
```

```py
print("data value counts():")
data["left"].value_counts()
```

```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```

```py
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```

```py
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```

```py
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```

```py
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

```py
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

```py
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

```py
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```

## Output:

<img width="1032" height="190" alt="image" src="https://github.com/user-attachments/assets/612077c9-24f8-49de-80cc-ee3354cdcd40" />

<img width="976" height="377" alt="image" src="https://github.com/user-attachments/assets/125d6c5e-360d-442c-b109-43a84697e5ac" />

<img width="1040" height="307" alt="image" src="https://github.com/user-attachments/assets/962f7bce-da9f-48f1-947e-387438ce6a38" />

<img width="560" height="137" alt="image" src="https://github.com/user-attachments/assets/dda075ae-ab66-4510-b5a9-bc16c38434dc" />

<img width="1045" height="186" alt="image" src="https://github.com/user-attachments/assets/8ece586b-c60c-4834-97d1-66a571b1d6ce" />

<img width="1023" height="192" alt="image" src="https://github.com/user-attachments/assets/a20938b2-e984-42bf-85af-b73528dcaa64" />

<img width="555" height="77" alt="image" src="https://github.com/user-attachments/assets/ae9df3bf-c375-4237-a075-590a389fb317" />

<img width="1030" height="100" alt="image" src="https://github.com/user-attachments/assets/8e2af904-af2a-4143-bd38-283bae8cadc4" />

<img width="802" height="591" alt="image" src="https://github.com/user-attachments/assets/b6e9e223-0f11-4ac6-af4c-12577598190d" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
