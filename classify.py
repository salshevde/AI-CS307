import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ------------- LOAD DATA -------------

data = pd.read_csv(
    "./new-thyroid.data",
    header=None,
    names=[
        "thyroid",
        "T3R",
        "Total Serum Thyroxin",
        "Total serum triiodothyronine",
        "	basal thyroid-stimulating hormone (TSH)",

        "Maximal absolute difference of TSH value",
    ],
)


# ------------- SPLIT DATA -------------
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=30
)
# ------------- TRAIN MODEL -------------

GNBclassifier = GaussianNB()
GNBclassifier.fit(x_train, y_train)


# ------------- ACCURACY TESTING -------------
y_test_pred = GNBclassifier.predict(x_test)

print(y_test)
print(y_test_pred)
print(f"Accuracy of the model: {accuracy_score(y_test,y_test_pred)*100}%")

# ------------- PREDICT USING MODEL -------------

prediction_data = input("Input filename for prediction: ")
x_pred = pd.read_csv(prediction_data,    header=None,
    names=[
        "thyroid",
        "T3R",
        "Total Serum Thyroxin",
        "Total serum triiodothyronine",
        "	basal thyroid-stimulating hormone (TSH)",
        "Maximal absolute difference of TSH value",
    ],).drop(["thyroid"],axis=1)
y_pred = GNBclassifier.predict(x_pred).map({1:"Normal Thyroid Function",2:"Hyperthyroid",3:"Hypothyroid"})
print(y_pred)
results = pd.concat([x_pred,pd.DataFrame(y_pred,columns=['THYROID PREDICTION'])],axis=1)
results.to_csv('predictions.csv',index=False)
