import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# *************************** TAHMİN *****************************
'''
Kategorik veriler üzerinde herhangi bir tahmin yapıldığında sınıflandırma olarak
Sayısal değerleri tahmin ederken ise tahmin yapıyoruz
Tarihe bağlı olduğunda zaman serisi analizi yapılıyor
Prediction ile forecasting arasındaki fark zaman ile alakalı. Forecasting geleceğe yönelik öngörülerdir, tahmin geçmiş verileri tahmin etmek için de kullanılabilir
'''

# VERI YUKLEME
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

# VERILERI OLCEKLENDIRME VEYA OLCEKLENDIRMEME ISTE TUM MESELE BU
'''
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
'''
BURADA VERİLERİ STANDARTLAŞTIRDIĞI ŞEKLİYLE MODEL OLUŞTURDU
# Help i açmak için tıkladıktan sonra ctrl+ı ya basılır
# Model için sklearn içindeki model ile ilgili kütüphaneleri eklemeliyiz
# Model İnşası lr.fit fonksiyonu ile yapılır
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
tahmin = lr.predict(X_test)
'''






# DOGRUSAL REGRESYON
'''
y = ax +b şeklinde model kurulur
a doğrunun eğimini b ise ne kadar kaydıralacağını gösteriyor
Hata miktarı doğru ile nokta arasındaki değerlerin toplamı 
'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

'''
Veri Görselleştirme
Veri görselleştirme için matplot kütüphanesi plt ismiyle kullanılır
Düzgün bir şekilde plot oluşturmak için bu değerleri indexlerine göre sıralamak gerekir
'''
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()
print("Linear R2 degeri:")
print(r2_score(Y, lin_reg.predict((X))))






# COKLU DOGRUSAL REGRESYON
'''
#                               DUMMY VARIABLE
Dummy - Kukla değişken Aynı anda 2 bilginin de veri kümesinde bulunması bir risk arzeder
Bazı ML algoritmaları Dummy değişkenlere karşı dayanıklı değildir
Cinsiyet, şehir gibi kategorik değişkenlerin olduğu durumlarda kullanılır
Örneğin kadınları araştırıyoruz, erkeklerin olduğu ve cinsiyet(E|K) değişkenleri aynı anda modele konulmaz
Kolonlar arasında bağlantı, kolerasyon olmaması için dummy variable düzgün seçilmelidir
Bir kişinin erkek olmaması kadın olması anlamına geliyor şeklinde çıkarımlar yapılmalıdır
'''
'''
#                               MODEL SECIMI
 1- Öncelikle tüm değişkenleri modele ekleyebiliriz
 2-Geri Doğru Eleme ile kolon eksilterek ilerliyoruz.
 Amacımız significance level olarak bilinen alpha değişkenine göre çıkarma işlemi yapılır
 P>SL ise en yüksek p değerine sahip olan değişken çıkarılır, P<SL ise değişken çıkarma işlemi durdurulur 
 3- İleri doğru seçimde p değeri en düşük olan modelde kalıyor
 Yukarıdaki p değeri karşılaştırması işlemi tekrarlanıyor
 4- Çift yönlü; yukarıdaki iki yönlü işlem de burada uygulanır. 2 sign. kullanılabilir ileri ve geri için ayrı
 Sisteme her birinin olan etkisi ayrı ayrı ölçülebiliyor
 5- Skor karşılaştırılması; kendi kriterini kendin belirlediğin zaman uygulanır.
 Kaç adet değişken var ise 
'''
veriler = pd.read_csv('veriler.csv')
print(veriler)
c = veriler.iloc[:,-1:].values
print(c)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(c)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)
#Sadece bir kolonu aldık

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection  import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s2,sonuc3,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
#### Modelling
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# Buradan öğrenilen(fit edilen) doğrunun formülü 6 boyutludur
y_pred = regressor.predict(x_test)


#                                        Boy tahmini
# S2'den boy kolonunu çekip train test kümelerine böldüğüm yapıya verecem Y değeri boy olacak
# Veri manipülasyonu işlemi
boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol, sag], axis = 1)
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)
y_pred = regressor2.predict(x_test)

'''
#                                    GERIYE DOGRU ELEME
 Başarı kriterlerinin belirlenmesi
 Modelin başarısını ölçmek için stats.model.formula.api çağırıyoruz
'''
import statsmodels.formula.api as sm
# B0 değerleri için aşağıdaki işlemler yapılır
# sütun olarak 1 çarpanı ekleyen np.append fonksiyonu
X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
# X_l ile bağımsız değişkenlerin oluşturduğu diziyi oluşturduk
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary())
#X5 değerinin p değeri 0.639 o yüzden bu değişkeni çıkarmalıyız

X_l = veri.iloc[:,[0,1,2,3,5]].values
# X_l ile bağımsız değişkenlerin oluşturduğu diziyi oluşturduk
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary())

X_l = veri.iloc[:,[0,1,2,3]].values
# X_l ile bağımsız değişkenlerin oluşturduğu diziyi oluşturduk
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary())





# POLINOMAL REGRESYON
'''
 Doğrusal değil polinomal. Bağımlı değişken olan Xler'in artık dereceleri var
 X1ler X2ler modelde ayrı ayrı şekillerde yer alabilir. Birbirleri ile çarpılan şekilde de olabilir
'''

veriler = pd.read_csv('maaslar.csv')
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#1 LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
'''
 Bu şekilde aslında lineer olmayan bir modeli eğitiyoruz
 Verilerdeki eğitim seviyesi ve maaşı iki boyutlu uzaya oturtacaz
 predict fonk kullanarak her bir xe karşılık gelen tahminleri yerleştirecez
'''
#2 NONLINEAR 2. DERECEDEN MODEL
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
'''
 PolynomialFeatures dan oluşturulan 2. dereceden bir obje
 Bu şekilde noktalara daha yakın geçiyor
 Dikkat edilmesi gereken şey tahmin değerini vermeden öncede polinomal features'a çevirmek gerekir(2,4 sen karar ver)
'''

#3 NONLINEAR 4. DERECEDEN MODEL
'''
 PolynomialFeatures 4 değerini verdikten sonra noktalara olan yakınlığını görebiliyoruz
 Polinom derecesini arttırdığımız zaman daha da noktalara yakınlaştı
'''
poly_reg3 = PolynomialFeatures(degree = 4)
# PolynomialFeatures dan oluşturulan 2. dereceden bir obje
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y)

#Doğrusal regresyona göre olduğunda 6.6 da normalden daha çok veriyor
print(lin_reg.predict(np.array([11]).reshape(1,-1)))
print(lin_reg.predict(np.array([6.6]).reshape(1,-1)))

# Polinomal Regresyon için aşağıdaki dönüşüm uygulanır
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[11]]))))
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[6.6]]))))

#Görselleştirme
#1
plt.scatter(X, Y, color = 'red')
plt.plot(x, lin_reg.predict(X), color = 'blue')

#2. DERECEDEN
plt.scatter(X,Y,color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
# Herbir x değeri için polinomal dünyaya çevir sonra çiz
plt.show()

#3 (4.DERECEDEN POLİNOM)
plt.scatter(X,Y,color = 'red')
plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
# Herbir x değeri için polinomal dünyaya çevir sonra çiz
plt.show()






# SUPPORT VECTOR REGRESSION
'''
 Sınıflandırma porblemi olarak biliniyor
 Doğrusal olarak birbirinden ayrılabilecek 2 sınıfı birbirinden ayırmaya yarar
 Amaç maksimum marjini elde etmek yani en geniş otoyolu elde etmek
 Hangi doğru bu iki kümeyi daha iyi ayırırın cevabı maksimum marjin değerine sahip olan anlamına gelir
 Eğriler çizmek de mümkün
 Bir marjin alanına en çok veri noktası alabilmek önemli olan
 Marjin miktarı EPSİLON değeri ile genişletilir veya daraltılır
 Aynı zamanda marjin değeri minimize de edilmelidir
 Farklı yaklaşımlı regresyon modelleri SVR içerisinde kullanılabilir
 SVR veriler üzerinde model oluştururken outlier verilere karşı dayanıklı değil. Dolayısıyla bu kodu kullanırken scaler kullanmalıyız
'''
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))
print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )






# DECISION TREE REGRESSOR
'''
 Öncelikle neye göre böldüğümüzü belirliyoruz
 Karar ağacı kökten yapraklara doğru giden bir yapıdır
 xy düzlemine oturtulan veriyi çeşitli sınırlarla bölerek karar ağacını oluşturuyoruz
 Kaç bölge varsa o sayı kadar yaprak düğümü(boş alan) vardır
 Her bölgeye o bölgenin bağımlı değişkenine ait ortalamasını yazıyoruz
 Son adımda bu ortalamarı ağaca yerleştiriyoruz(boş alanlara)
 Öğrenme sürecini bitirdik - test aşamasına geçilebilir
 Belirtilen uzayın tüm alan için aynı değer(ortalama). Daha detaylı bilgi için ağaç daha da budaklanacak anlamına geliyor
'''
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')
plt.show()
print(r_dt.predict(np.array([11]).reshape(1,-1)))
print(r_dt.predict(np.array([6.6]).reshape(1,-1)))
# Tüm değerler aynı gruba indiriliyor. Bunun sebebi gelen aralığa göre indirgenir
# Karar ağaçları bağımlı değişkendeki aynı sayıları döndürür

print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)) )



# RANDOM FOREST REGRESSION
'''
 Essemble Learning(Kollektif Öğrenme); birden fazla algoritma aynı anda kullılarak daha başarılı bir sonuç çıkarabailir
 Bu algoritmaları birlikte kullanılarak hata azaltılabilir
 Random forest birden fazla karar ağacının aynı veri üzerinde çizilmesi ve kullanılması anlamına geliyor
 Veri kümeyi alt küçük parçalara bölerek birden fazla karar ağacı oluşturacak
 Rassal orman ile çoğunluğun oyunu kullanıyor ve çoğunluğun tahminini kullanıyor(Sınıflandırma işlemlerinde kullanılır)
 Training aşamasında veri kümesini parçalara bölüyoruz 
 Verinin artması durumunda başarı düşebilir. 1- sonuclar yanlış çıkabilir overfitting(ezberlemeye) gidebilir. 2- Çok fazla dallanması ve hesaplama zamanının uzaması anlamına geliyor
'''
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X,Y)
print(rf_reg.predict(np.array([6.6]).reshape(1,-1)))
# Birden fazla dec.tree olduğu için ortalamalar döndürüyoruz. Dolayısıyla tahmin aşamasında orjinal veriler dışında değerler döndürebilir

'''
 n_estimators ile kaç tane dec.tree çizileceğini yazıyoruz
 verinin 10 farklı parçası ile algoritması rf_reg objesinin içine koyulacak
 ile eğitim işlemi yapıyorum. ikisi de numeric sıkıntı yok
 tahmin için kullanılan RandomForestRegressor. sınıflandırmaya geçtiğimizde yine randomforest orjinal algoritmasını kullanacağız
'''

print(rf_reg.predict(6.6))
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y, color='red')
plt.plot(x,rf_reg.predict(X), color = 'blue')
plt.plot(x,rf_reg.predict(Z), color = 'green')
plt.plot(x,r_dt.predict(K), color = 'yellow')
plt.show()

print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)) )
print(r2_score(Y, rf_reg.predict(K)) )
print(r2_score(Y, rf_reg.predict(Z)) )






# RSQUARE DEGERLERI
'''
 RKare gerçek değerler ile tahmin ettiğimiz değerler arasında bir ilişki veriyor
 Farklı algoritmaların karşılaştırılmasında Rkare den yararlanılır
 Rkare ne kadar büyük ise o kadar başarılı bir model kurulmuştur
 R kare modelin gelişip gelişmediğini tam olarak göstermeyebilir(modele 1 değişken daha eklendiğinde R2 değerini artırır ama azaltmaz)
 Bir çözüm olarak Adjusted RSquare değeri kullanılır
'''
print('----------------')
print("Linear R2 degeri:")
print(r2_score(Y, lin_reg.predict((X))))


print("Polynomial R2 degeri:")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X)) ))


print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )


print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)) )

print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)) )




