import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# ------------- LOAD DATA -------------
with open('./columns.names','r') as file:
    columns = [line.strip().split(':')[0] for line in file.readlines() if line.strip().split(':')[0]]
# fix
data = pd.read_csv('./hypothyroid.data',header=None, names = columns)

# ------------- PREPROCESS DATA -------------
data.replace('?',pd.NA,inplace =True)
data.ffill(inplace=True)

continuous_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

# Convert continuous columns to float, handling NaN values
for col in continuous_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to float and coerce errors

data[continuous_cols].fillna(data[continuous_cols].mean(), inplace=True)
# Convert binary categorical columns to integers
binary_cols = [
    'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'thyroid_surgery', 'query_hypothyroid', 'query_hyperthyroid',
    'pregnant', 'sick', 'tumor', 'lithium', 'goitre', 'TSH_measured',
    'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured'
]

for col in binary_cols:
    data[col] = data[col].map({'f':0,'t':1})
# ------------- SPLIT DATA -------------
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
print(X)
x_train, x_test, y_train,y_test = train_test_split(X,Y, test_size = 0.3,random_state=30)

# ------------- TRAIN MODEL -------------

GNBclassifier = GaussianNB()
GNBclassifier.fit(x_train,y_train)


# ------------- ACCURACY TESTING -------------

y_test_pred = GNBclassifier.predict(x_test)
print(f"Accuracy of the model: {accuracy_score(y_test,y_test_pred)}%")
print(y_test)
print(y_test_pred)

# ------------- PREDICT USING MODEL -------------



