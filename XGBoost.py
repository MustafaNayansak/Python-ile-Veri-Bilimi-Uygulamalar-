
#1. kutuphaneler
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")

#veri on isleme
X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X = X[:,1:]


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=0)


# XGBoost 
'''
XGBoost büyük ölçekteki verileri hızlıca analiz edebilir
XGboost ile problem ve modelinin yorumunu mümkün
Makine Öğrenmesi yarışmalarında öncü algoritmalardan bir tanesidir
'''

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
  
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)