#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('Wine.csv')
#pd.read_csv("veriler.csv")


#veri on isleme
X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values
 

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection  import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# PCA ile elimizdeki 13 kolonu 2 kolona indirgeyip müşteri segmentasyonu yapacağız

# PCA 

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# PCA dönüşümünden önce gelen LR(13 değişken)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# PCA dönüşümünden önce gelen LR(2 değişken)
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)


# Tahminler
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
# Actual / PCA olmadan çıkan sonuc
print("Gerçek / PCAsız")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ACtual / PCA sonrası çıkan sonuc
print("Gerçek / PCA ile")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm)

# PCA sonrası / PCA öncesi
print("PCAsız / PCA ile")
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm)



#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
veriler = pd.read_csv('Wine.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
#actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test,y_pred)
print(cm)

#actual / PCA sonrası çıkan sonuç
print("gercek / pca ile")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

#PCA sonrası / PCA öncesi
print('pcasiz ve pcali')
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)



# Linear Discriminant Analysis(LDA)
# PCA gibi bir boyut indirgeme algoritmasıdır
# Sınıflar arasındaki ayrımı önemser ve maksimize etmeye çalışır
# PCA unsupervised - LDA supervised özelliklidir
# Sadece verileri en iyi gösteren şekildedeğil, sınıfları güzelce ayırmak önemlidir
# LDA için veriler etiketlenmiş olmalıdır

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
# PCAda xtrain2 fit.transform etmiştik. LDA de y_traini de fit.transform edecek. Supervised öğrenme olduğu için
X_test_lda = lda.transform(X_test)


# LDA dönüşümünden sonra 
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

# LDA verisini tahmin et 
y_pred_lda = classifier_lda.predict(X_test_lda)

# LDA Sonrası ve Orijinal
print('LDA sonrası / Orijinal ')
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)


# PCA sınıfları gözetmediği için 1hata oldu
# LDA öyle güzel sınıflandırdı ki yine 100 de 100 başarı elde etti(sınıf farkını gözetti)

