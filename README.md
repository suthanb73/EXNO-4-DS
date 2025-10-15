# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
Name : suthan b
Register Number : 25018310
```

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")

df.head()
df.dropna()
```
![image](https://github.com/user-attachments/assets/ff7841c3-ea9d-4a2d-b2d5-984a8ef9e1dd)

```
max_vals=np.max(np.abs(df[['Height']]))
max_vals
max_vals1=np.max(np.abs(df[['Weight']]))
max_vals1
print("Height =",max_vals)
print("Weight =",max_vals1)
```
![Screenshot 2024-10-08 160759](https://github.com/user-attachments/assets/6cec529f-5068-4bd6-9cbe-76b7caf25c44)
```
df1=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![Screenshot 2024-10-08 160906](https://github.com/user-attachments/assets/7735d008-2707-4e0e-b5fa-caeb2be7e327)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-10-08 160957](https://github.com/user-attachments/assets/6d8ae21c-fcf5-45b4-af37-000af98b9f8f)

```
from sklearn.preprocessing import Normalizer
df2=pd.read_csv("/content/bmi.csv")
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-08 161054](https://github.com/user-attachments/assets/2fdead27-5061-487a-9e6e-2ac7f84dc1aa)

```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![Screenshot 2024-10-08 161325](https://github.com/user-attachments/assets/331bd452-a95b-495b-94c3-5c81affec7eb)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![Screenshot 2024-10-08 161453](https://github.com/user-attachments/assets/4e8fab38-250a-40a7-8302-cb05cf205e30)

## FEATURE SELECTION SUING KNN CLASSIFICATION:

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-10-08 161612](https://github.com/user-attachments/assets/48630cbd-57b3-44dd-8660-a7ddeca3ba74)

```
data.isnull().sum()
```
![Screenshot 2024-10-08 161818](https://github.com/user-attachments/assets/ccf972f0-a972-43d2-a2bb-6d1d64c29a8b)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-08 161918](https://github.com/user-attachments/assets/aa82db06-6d1b-4ed0-ac4a-810d381d12b9)

```
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-10-08 162013](https://github.com/user-attachments/assets/f8d1610d-4faa-47b8-8391-c764102e0262)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```
![Screenshot 2024-10-08 162123](https://github.com/user-attachments/assets/109845b5-174e-4ba1-8ca5-546a47ac00ec)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-10-08 162206](https://github.com/user-attachments/assets/dbb61dd7-6ea0-486e-8cf0-b9111d9673f4)

```
data2
```
![Screenshot 2024-10-08 162252](https://github.com/user-attachments/assets/bd797da5-d0eb-42f1-b71a-be691f3676f3)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-10-08 162339](https://github.com/user-attachments/assets/2c5cf57d-9dd6-4ad4-b61d-416a747e89b6)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-10-08 162437](https://github.com/user-attachments/assets/b4ec0bfb-7bfe-41b9-878a-aa4a294625e2)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/a5f40b82-52b0-4ce8-9756-8ea04f222c6c)

```
y=new_data['SalStat']
print(y)
```
![Screenshot 2024-10-08 162745](https://github.com/user-attachments/assets/e2369ab4-5449-476f-b74d-e7d25b12e580)

```
x=new_data[features].values
print(x)
```
![Screenshot 2024-10-08 162946](https://github.com/user-attachments/assets/e6ee0a5e-bb06-49bd-8db1-cdb30346d871)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![Screenshot 2024-10-08 163032](https://github.com/user-attachments/assets/63f0f7c7-b2a2-4a0a-ae15-07ca457c813c)

```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![Screenshot 2024-10-08 163114](https://github.com/user-attachments/assets/a5d462f3-5c48-4d44-bb1a-eddda40170c3)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)

X=df[['Feature1','Feature3']]
y=df['Target']

selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)

selected_feature_indices = selector.get_support(indices=True)

```

![Screenshot 2024-10-08 234436](https://github.com/user-attachments/assets/f58d4f84-ab62-4ed1-9614-c67a082f514f)


# RESULT:
       Thus, Feature selection and Feature scaling has been used on the given dataset.
