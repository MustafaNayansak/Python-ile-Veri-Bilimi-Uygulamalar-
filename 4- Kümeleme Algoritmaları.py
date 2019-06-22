# ******************** KUMELEME ********************
# Ne aradığını bilmiyorsan ve verileri bazı segmentleri veya kümeleri tanımlamaya çalışıyorsan ilk bakacağın problem tipi
# Gözetimsiz öğrenme yöntemlerinden bir tanesidir
# Aynı tipteki verileri birbirinden ayırmaya çalışıyoruz
# Herhangi bir ön bilgi verilmiyor sınıfları kendisi belirliyor
# Müşteri-Pazar Segmentasyonu, Sağlık ve Görüntü işleme alanlarında kullanılır
# Müşteriyi doğru ürünle birleştirmek, müşteriye özel kampanyalar
# Amaç kümeler arası mesafe minimum, kümeler arası mesafenin maksimum olması üzerine kuruludur


# KMEANS CLUSTERING
'''
1 Kam küme olcağı parametre olarak seçilmeli
2 Rastgele k tane merkez seçer ve merkez noktalarını kaydırarak kümeleme işlemi yapıyor
3 Yeni eklenen noktalar ile noktalar dinamik olarak yer değişiyor
4 Noktalar kararlı hale getirince kümeleme işlemi bitmiş oluyor
5 KMeans ile atanan noktaların dağılımı bazen yanlış atanabilir

 Kmeans++ algoritması random problemini; 
1 noktaları dağıtıyor ve mesafeleri(DX) hesaplıyor. 
2 DX değerleri her bir nokta için hesaplandıktan sonra Yeni noktaları mesafenin karesini olasılık alarak bulur(DX^2)

 Küme sayısına nasıl karar verilir
 Gözetimsiz öğrenme olduğu için herhangi bir bilgi olmadığı için optimum nokta;
 WCSS her bir cluster için o kümedeki elemanların merkeze olan uzaklıüğının karesi toplanır(her kümeler için yapılır ve toplanır)
 WCSS'de dirsek noktası olarak bir nokta seçilir(eğim dğeişimi) ve ordaki k değeri o değerdir

 sklearn clustering burada kümelemenin karşılaştırıldığı bir yer var. Farklı problemli veri türlerinde hangi algoritmanın daha uygun olacğaını görebilirsin
 init(random) rastgele olduğu için noktaları istediğimiz gibi dağıtmayabilir 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')
print(veriler)

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)
print(kmeans.cluster_centers_)
# hacim ve maaş cinsinden orta noktaların koordinatlarını verdi

# k için optimum değeri bulma
sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)    
#WSS değerleri
 
plt.plot(range(1,11), sonuclar)
# 4 güzel- 2 de olabilir(Dirsek noktası)

    





# HIYERARSIK KUMELEME(HIERARCHICAL CLUSTERING) 
'''
 İki farklı şekilde çalışır.(AGGLOMORATİVEaşağıdan yukarıya ve DİVİTİVEyukarıdan aşağıya şeklinde)
 Her veri noktası bir küme, en yakın komşuları birleştiriyoruz, tüm uzay tek bir bölüte dönüşene kadar devam ediyor
 Mesafe ölçümü metrik ölçüm yöntemleri ile ölçülebilir - Referans noktaları olarak Enyakın,Enuzak, Ortalama, Ağırlık merkezlerine göre 
 Dendogram ile kümeleme adımlarını mesafelere doğru orantılı olarak görebiliyoruz
 Kullandığımız yönteme göre kümelemeler değişkenlik gösterecektir
 Her yöntem farklı verilerde farklı şekillerde çalışır
 WARDS method, cluster içindeki mesafelerin karelerinin toplamından oluşan yöntem ile 1. kümenin 2. kümenin ve 1.2 ortak kümenin oluşan WCSS toplamlarıdır
 Dendogram ile kaç cluster isteniyorsa nerden böleceğimizi söylüyor
'''

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

# cluster.hierarchy için scipy kütüphanesini kullanıyoruz
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()
# k nın 2 olması en uzun mesafeyi oluşturuyor
# 3 ile 4 arası çok kısa mesafe olduğu için 2 veya 4 alınabilir

    
    
    
    
    
    
# BIRLIKTELIK KURALI (ASSOCIATION RULE MINING)
'''
 Support : a varlığını içeren eylem sayısı / Toplam eylem sayısı(TOPLUMDA NE KADARININ BU EYLEMİ YAPTIĞI)
 Confidence : a ile b varlığını(bağlı eylemler) içerenler / a varlığını içeren eylemler
 Lift : conf(a,b) / support(b) || lift > 1 ise a alanlar b alıyor anlamına geliyor < ise olumsuz etkiliyor anlamındadır
 Apriori frekans sayarak ilerler
 ARM in yönlendirilmiş versiyonları A>B'ye olan yön önemlidir
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#Apriori Algorithm
veriler = pd.read_csv('sepet.csv', header =None )
t = []
#listeler listesi
# apriori bizden bu listeyi istiyor
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])
    
from apyori import apriori
kurallar = apriori(t, min_support=0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)
# support çok ürün varsa düşük ver
print(list(kurallar))

# kıyma alanlar baharat da alıyor. Conf ve lift değerleri veriyor. Normal de bu ürünleri alınma ihtimalini bir de birlikte alındıklari ihtimali inceliyor. alındıktan sonra 3.2 misli artıyor şeklimnde yorumlanabilir
# Birliktelik kurallarında en çok kullanılan apriori algoritmasına ek olarak ECLAT algoritması vardır
# ECLAT daha hızlı çalışır
# Apriori item bazlı gidiyor önce ürünlere bakıyor, eclat da ürünlerin geçtiği transaction'a bakılır







    
    
    
    