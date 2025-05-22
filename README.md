# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
Program to implement the SVM For Spam Mail Detection..

Developed by:Nandhini S

RegisterNumber: 212224230174 

```
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
     print('Result output')
     result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```

## Output:

## Encoding:

![Screenshot 2025-05-22 105703](https://github.com/user-attachments/assets/1efa8884-113d-435b-81a6-971b0ba8ca91)

## Head():
![Screenshot 2025-05-22 105753](https://github.com/user-attachments/assets/08d70d44-bf14-483b-ae8b-071b0af66b68)

## Info():

![Screenshot 2025-05-22 105841](https://github.com/user-attachments/assets/ed7901b8-0450-4f21-820b-f72e16b2b571)
## isnull().sum():
![Screenshot 2025-05-22 105935](https://github.com/user-attachments/assets/045a8022-9d9b-4703-b5df-5ce906a33ad9)
## Prediction of y:

![Screenshot 2025-05-22 110007](https://github.com/user-attachments/assets/b3cffdfa-3794-4d70-9329-b42c557e018e)
## Accuracy:
![Screenshot 2025-05-22 110039](https://github.com/user-attachments/assets/8e2832c0-76a1-4c24-adf2-a0525f84426b)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
