<H3>ENTER YOUR NAME : PREM R</H3>
<H3>ENTER YOUR REGISTER NO : 212223240124</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 19-08-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv("Iris.csv")
data
data.head()
X=data.iloc[:,:-1].values
X
y=data.iloc[:,-1].values
y
data.isnull().sum()
data.duplicated()
data.describe()
data = data.drop(['Id','SepalLengthCm','SepalWidthCm'], axis=1)
data.head()
scaler=MinMaxScaler()
# Exclude the 'Species' column before scaling
numerical_data = data.drop('Species', axis=1)
df1 = pd.DataFrame(scaler.fit_transform(numerical_data), columns=numerical_data.columns)
print(df1)
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train
X_test
print("Lenght of X_test ",len(X_test))
```
## OUTPUT:

<img width="584" height="346" alt="image" src="https://github.com/user-attachments/assets/a8ac8dea-e852-4e05-9273-0ddc70e8d642" />
<br>
<img width="620" height="184" alt="image" src="https://github.com/user-attachments/assets/009c9c18-e38e-4684-9ad9-ff19322adea3" />
<br>
<img width="557" height="546" alt="image" src="https://github.com/user-attachments/assets/e0776476-243c-4886-9633-d1ba53a8daaa" />
<br>
<img width="669" height="678" alt="image" src="https://github.com/user-attachments/assets/832ea199-99f4-4f19-a56f-1cc59575e556" />
<br>
<img width="227" height="236" alt="image" src="https://github.com/user-attachments/assets/8e614954-ebac-4c1a-b0e5-1afaa0cb2686" />
<br>
<img width="247" height="381" alt="image" src="https://github.com/user-attachments/assets/ddb4fbbb-312f-49a3-b061-4cd2aaec6c9c" />
<br>
<img width="646" height="259" alt="image" src="https://github.com/user-attachments/assets/b502073d-6eb7-4562-88b5-8b634aad5d39" />
<br>
<img width="388" height="192" alt="image" src="https://github.com/user-attachments/assets/e92e80a8-66b2-4db1-a236-e014bcada998" />
<br>
<img width="441" height="223" alt="image" src="https://github.com/user-attachments/assets/808aff04-063b-4005-9482-f7d765d235e4" />
<br>
<img width="511" height="726" alt="image" src="https://github.com/user-attachments/assets/1a5fd685-cd27-4890-8783-b296ccab5f58" />
<br>
<img width="418" height="449" alt="image" src="https://github.com/user-attachments/assets/e130ddf8-7738-4683-b5f4-48a083a684c5" />
<br>
<img width="262" height="41" alt="image" src="https://github.com/user-attachments/assets/f1cb9c63-9751-47d1-ba69-9fa7607b4e3c" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


