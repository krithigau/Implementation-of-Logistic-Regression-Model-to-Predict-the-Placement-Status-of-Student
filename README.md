# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Data preprocessing:
3. Cleanse data,handle missing values,encode categorical variables.
4. Model Training:Fit logistic regression model on preprocessed data.
5. Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6. Prediction: Predict placement status for new student data using trained model.
7. End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KRITHIGA U
RegisterNumber:  212223240076
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
print("\nOpening File\n")
print(data.head())

data1=data.copy()
print("\nDroping File\n")
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
print(data1.head())

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
print(data1)

x=data1.iloc[:,:-1]
print(x)

y=data1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)

print("\nClassification report\n")
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


print("\nTesting model\n")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
Opening File:

![Screenshot 2024-09-23 114014](https://github.com/user-attachments/assets/c1d566ed-d2c2-46f4-985e-72a2713c0639)

Droping File:

![Screenshot 2024-09-23 114113](https://github.com/user-attachments/assets/b4389b25-0c22-4a7a-8bd7-ea70c5a8808f)

Duplicated():

![Screenshot 2024-09-23 114218](https://github.com/user-attachments/assets/103b50b3-754a-4aab-a85c-1ce26ddd33cd)

Label Encoding:

![Screenshot 2024-09-23 114323](https://github.com/user-attachments/assets/8eefe3a7-9ca0-4ef7-9a9c-1524aa76907a)

Spliting x,y:

![Screenshot 2024-09-23 114422](https://github.com/user-attachments/assets/7a6dfbac-8f56-42af-8737-2fd2211c1d0c)
![Screenshot 2024-09-23 114510](https://github.com/user-attachments/assets/409dd0b7-f214-43ce-954d-824783417b9e)

Prediction Score:

![Screenshot 2024-09-23 114715](https://github.com/user-attachments/assets/86cbe1be-e16a-4bc5-bd53-e0b4edf925a4)

Testing accuracy:

![Screenshot 2024-09-23 114720](https://github.com/user-attachments/assets/ce03adb3-b549-4cab-a4a4-e31d19d1b4fc)

Classifictaion Report:

![Screenshot 2024-09-23 114729](https://github.com/user-attachments/assets/ea6f171d-41d7-4c60-8bb5-344f9bb4b8eb)

Testing Model:

![Screenshot 2024-09-23 114734](https://github.com/user-attachments/assets/3dd7dc90-f3fa-4b57-980f-1d79d79fd574)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
