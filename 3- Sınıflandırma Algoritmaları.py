import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
#fit eğitme - transform ise o eğitimi uygulama. Fit_transform x ten öğren transform et ve yeniden öğrenme anlamına geliyor







#  LOJISTIK REGRESYON

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #egitim

y_pred = logr.predict(X_test) #tahmin
print(y_pred)
print(y_test)

#karmasiklik matrisi
cm = confusion_matrix(y_test,y_pred)
print(cm)






#  K EN YAKIN KOMSU ALGORITMASI(KNN)
'''
 Veriler 2 boyutlu uzaya oturtulduktan sonra yeni bir gözlem eklendiğinde komşuları ile olan ilişkilerine(uzaklıklarına) göre hangi sınıfa ekleneceği kararlaştırılır
 Mesafeleri kullanırken oklid mesafesini kullanıyor. Boy kilo arasındaki uzaklığı ölçmek için farklı yöntemler kullanılabilir
 Lasy learning ile herhangi bir öğrenme yapmıyor yeni bir örnek geldiğinde nasıl sınıflandıracağına bakmak için komşularını inceliyor
 Eager learning bölgeleri önceden hazırlıyor ve bu verileri unutuyor daha sonra bir örnek geldiğinde onu nereye sınıflandırdığına bakarak öğreniyor
 sklearn.knn yazarak kaynaklara ulaşabilrsin
 Sayısal tahmin yapmak içinde nn algoritması(regresyon) kullanılabilir
 metric= minkowski algoritması kullanılır P=2 için default olarak öklid mesafesi kullanılır
'''
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)
'''
# outlier'lar sistemin yanlış çalışmasına sebep oluyor
# knn de kümelerin içinde küçük adacıklar oluşabilir
'''






# 3. DESTEK VEKTOR SINIFLANDIRICISI(SVC/SVM)
'''
 Sınıflandırma için çıkmış, 2 grubu birbirinden ayırmak için kullanılır
 MAkine öğrenmesi bu iki grubu ayıran çizgiyi öğrenir 
 Çizgilerin hangisi en iyi sınıflandırmayı yapar. Marjin'i maksimize etmek iki sınıfı birbirinden mükemmel bir şekilde ayırmak demek
 Marjinlerin üzerinden geçtiği noktalar support vektör adı verilir
 Regresyonda bu aralığa düşen maksimum örnek sayısı, sınıflandırmada ise bu aralığa hiçbir noktanın düşmemesi istenir
 sklearn svm şeklinde aratarak kullanımını inceleyebilirsin
 Farklı farklı kernel ile denemeler yapabiliriz
 multiclass classification = one against one ile her sınıfı kendi arasında sınıflandırmaya çalışır(3 sınıf 3 farklı sınıflandırma problemi olarak ele alır)

'''
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
# sklearn den kernel ları değiştirerek('poly') farklı sonuçlar elde edebilirsin(parametre optimizasyonu)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)
'''
 Doğrusal olarak ayrıştırılamayan problemler için Çekirdek(kernel) hilesi
 Soft marjin ve hard marjin ler verinin bu marjin içine girip giremeyeceğini belirtir. 
 Bazı örneklerde doğrusal olarak marjin ayarlamak yerine 
 Eğer iç içe geçen 2 grup var ise. Ortadan bir tane nokta belirlenir ve o noktaya olan uzaklıklar diğer noktalar ile çarpılarak bir derinlik veya yükseklik oluşturulabilir(3. boyut ekleme)
 Derinlik veya yüksekliği oluşturduktan sonra SVM algoritması ile bir kesit ile 2 gruba bölmüş oluyoruz
 Doğrusal olarak ayrıştırılması imkansız olan iki sınıfı bu şekilde ayırabiliriz
 Gauss hesabına göre sigma değerinin büyüklüğüne göre noktalar arasındaki uzaklıkla hespalanabilir
 Çoklu kernel da kullanılabilir. 3 farklı noktadan da bu işlem yapılabilir
'''






# 4. NAIVE BAYES
'''
 Koşullu olasılığa dayanır
 Bayes'in bize sağladığı imkan: olasılıklar değiştirilebilir  
 Dengesiz veri kümelerinde çalışabilir
 Sınıflandırma işlemi sırasında olasılıklarda yararlanır
 KNN metrik kulanırken naive bayes tüm dağılıma bakarak her bir bireyin özellikleri ile basitleştiriyor(ilkelliyor)
 Naive bayes lasy learning ailesindedir
 EAGER LEARNING : Veri daha gelmeden önce tüm değerler için olasılıkları bir yerde bekletiyor ve yeni gelen örneği direk olarak kullanabilir
 Big data ile lasy learning kullanımı daha da arttı
 sklearn üzerinden kullanımını incele
 Gaussian-Multinominal-Bernoulli dağılımı üzerinden NAIVE bayes uygulanır
 Tahmin etmek istediğimiz veri sürekli bir değişken ise GAUSSIAN - Nominal değerlere integer sayılar veriyorsan Multinominal - 1 veya 0 gibi nominal değerleri alıyorsa Bernoulli NAIVE BAYES kullanılır
'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)






# KARAR AGACLARI(DECISION TREES)
'''
 Bölgeler oluşturuyor ve Her bölgedeki öğrelerin taşıdığı özellikleri taşır
 Çoğunluk hangi sınıfsa o bölgenin durumu ona göre seçilir
 Veya Daha da bölünerek tamamen homojen olunana kadar bölmeye devam edilebilir(çok detaylı bölersek overfitting olabilir)
 Karar ağacının nasıl çizileceği "entropi" kavramı üzerinden yapılır
 Kolonun ürettiği enformasyon değerine bakılır. Verinin ne kadar iyi dağıldığı çok önemli. 
 En fazla informasyon kazandıran değişken ile kökden gövdelenir
 Yes ve No 'ları pure saf bir şekle getirdikten sonra soru sorma işlemi biter. purenode
 sklearn decisiontreeclassifier incele
 Entropi - Giny indexi arasında çok bir fark yok araştırabilirsin
'''
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)






# RANDOM FOREST
'''
 sklearn random forest
 default ta giny indexini kullanıyor
 rassal ağacın kaç tane oluşturacağı n_estimators
'''
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)






# CONFUSION MATRİX
# Confusion Matrix
# ZeroR algoritması - Çoğunluk olan örneği sınıflandırma sonucu olarak kullan(eline gelen örneklerde hangi sınıf daha fazla varsa o sınıfa ata)
# ZeroR algoritması Accuracy değerine bakacak olursak yükseltir. Ama hiçbişey öğrenmiyor ve hiç sınıflandırma yapmıyor
# Accuracy değerine bakarak sınıflandırma algoritmalarını değerlendirmek doğru değildir
# Bir sınıflandırma algorimasını başarılı kabul etmemiz için bir alt çizgiye ihtiyaç vardır.
# Bazı konular için TRue negatifleri bulabilmek bazıları için TPleri bulabilmek önemlidir
# Seçilen algoritma ZeroR algoritmasını geçemiyorsa bu algoritmayı kullanmaya gerek yok

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)

# ROC Curve 
# TP VE TN kavramlarını iyi bilmek gerekir
# Değişik sınıflandırma algoritmasını roc üzerine oturttuğumuz zaman algoritmaları karşılaştırabiliyoruz
# Accuracy e göre karşılaştırmak tehlikeli olduğundan ROC üzerinden fikir sahibi olabiliriz
# Kitle arttıkça accuracy değerinde artış beklenir


    

