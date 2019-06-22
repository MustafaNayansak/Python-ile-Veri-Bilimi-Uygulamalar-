#1. KUTUPHANELER 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
Pandas kütüphanesini çağırıyor pd. dediğimizde kütüphanein altındaki fonksiyonlara ve özelliklere erişiyoruz
DF ler oluşturmak ve bu verilere düzgün bir şekilde ulaşabilmek için kullanılan kütüphane
Büyük sayılar ve hesaplama
'''




#2. VERI ON ISLEME



# VERI YUKLEME
veriler = pd.read_csv('Churn_Modelling.csv')
'''
pd.read_csv("C:\\users\..\veriler.csv") şeklinde de kullanılır
CSV(Comma Separated Value)
''' 



# VERILERI ATAMA
boy = veriler[['boy']]
# print(veriler.iloc[:,0:1])
print(boy)
boykilo = veriler[['boy','kilo']]
print(boykilo)
# print(veriler.iloc[:,1:2])



# ESKIK VERILER  
'''
Sayısal veriler için : Kolonun ortalama değeri yazdırılabilir
sci-kit learn makine öğrenmesi alet kutusu tarzı bişey
Pandas da eksik veriler NaN olarak gösterilir
Axis parametresi ile satır mı sütun basında mı ortalama alınacağını söyler
Sayısal olmayan verilerde imputer fonksiyonu çalışmaz dolayısıyla sayısal verileri seçiyoruz
'''
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)



# ENCODING :  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)



# DATAFRAME OLUSTURMA
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index=range(22), columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)



# TEST - TRAIN OLARAK IKIYE BOLME
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)



# VERILERIN OLCEKLENMESI
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)







    
    

