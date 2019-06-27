import seaborn as sns 
planets = sns.load_dataset('planets')
planets.head()

# Veri Setini Kopyaladık
df = planets.copy()

#Verinin yapısal bilgileri için
df.info()

# Değişkenlerin türleri için 
# Bazen değişkenler içinde yer alan - gözlemler değişkenin kategorik olarak algılanmasına yol açabilir
df.dtypes


# Kategorik değişkenin tipini değiştirebiliriz, hız için(object-char)
import pandas as pd 
df.method = pd.Categorical(df.method)
df.dtypes

df.head()
df.tail()

df.shape

#Değişken isimleri için
df.columns

# Describe fonksiyonu sadece sayısal değerler içindir
df.describe()
df.describe().T

df.nunique()

df.isnull().values.any()

# Tüm değişkenler kaçar tane eksik var
df.isnull().sum()

# DEğişkenlerin içinde yer alan tüm eksik değerlere 0 atama
df['orbital_period'].fillna(0, inplace = True)
# inplace parametresi ile df değişkenine bu işlemi atadı
df.isnull().sum()

# Eksik gözlemlerin yerine o değişkenin ortalama değerlerini eklemek için
import numpy as np
df['mass'].fillna(np.mean(df.mass), inplace = True)
df.isnull().sum()




# KATEGORİK DEĞİŞKENLERİN ÖZET İSTATİSTİKLERİ

kat_df = df.select_dtypes(include = ['category'])

kat_df.head()

# Değişkenin içerisindeki tüm sınıflar için
kat_df.method.unique()

# Sayıya erişmek için
kat_df['method'].value_counts().count()

# 10 adet sınıf var ama kaçar tane geçtiğine bakmak için
kat_df['method'].value_counts()

# Nominal olan sınıfların frekansları için histo 
df['method']. value_counts().plot.barh()

# Veri seti ordinal ise(rütbeler gibi) 
# Nominal değişkeni ordilnal değişkene çevirmek istediğimizde
from pandas.api.types import CategoricalDtype
kat_df = kat_df.astype(CategoricalDtype(ordered = True))

kat_df.method.head()



# SÜREKLİ DEĞİŞKENLERİN ÖZET İSTATİSTİKLERİ
df.head()
df_num = df.select_dtypes(include = ['float64','int64'])

df_num.describe().T

# Tek bir değişkenin istatistiklerine ulaşmak için
df_num['mass'].describe()

df['distance'].plot.hist(bins = 25)




# DAĞILIM 
##  BARPLOT

import seaborn as sns
diamonds = sns.load_dataset('diamonds')
df = diamonds.copy()
df.head

df.shape
df.dtypes
# sürekli değişkenler ve 3 tane de object var bunları kategorik olarak değiştirmeliyiz
df.describe().T

# değişkenleri categorik sıralı ve kategorik hale getirdik
from pandas.api.types import CategoricalDtype
df.cut = df.cut.astype(CategoricalDtype(ordered = True))
df.color = df.color.astype(CategoricalDtype(ordered = True))
df.clarity = df.clarity.astype(CategoricalDtype(ordered = True))

# Sıralamanın doğru olup olmadığı...
df.cut.head(1)
# Gerçek hali fair < good < very good < premium < ideal
# Sıralı ölçeğe getirme işlemi yanlış gerçekleşmiş
df.color.head(1)
df.clarity.head(1)
# Dönüşümleri yaptık ancak sıralı ölçeğe çevirme işlemlerinin tümü hatalı


# Ordinal değişkenleri kendi istediğimiz sırada sıraladık
cut_kategoriler = ['Fair','Good','Very Good','Premium','Ideal']
df.cut = df.cut.astype(CategoricalDtype(categories = cut_kategoriler,ordered = True))
df.cut.head(1)


# Kategorik Değişkenler için BARPLOT
df['cut'].value_counts().plot.barh().set_title('Kategırik Değişken Görselleştimek için Barplot ')

(df['cut']
    .value_counts()
    .plot.barh()
    .set_title('Kategırik Değişken Görselleştimek için Barplot '))

(df['clarity']
    .value_counts()
    .plot.barh()
    .set_title('Kategırik Değişken Görselleştimek için Barplot '))


sns.catplot(x = 'cut', y = 'carat', data = df)
sns.catplot(x = 'cut', y = 'price', data = df)
# Bu grafikte değişkenliği açıklamak adına çok iyi bir bilgi yok
sns.catplot(x = 'cut', y = 'price', hue = "color", data = df)

#ayar verelim
sns.catplot(x = 'cut', y = 'price', hue = "color", data = df)


## HISTOGRAM
df.head()
(df['price']
.plot
.hist(bins = 100))
# Burada bir çarpıklık gözlemlenebilir
sns.distplot(df.price, kde = False)
# Yoğunluk grafiği için;
sns.distplot(df.price)
sns.distplot(df.price, hist = False)

# Yoğunluk grafiğinin altını doldurmak için  
sns.kdeplot(df.price, shade = True)
# Histogram tek başına çok az bilgi taşıyor
# Burada price değişkennin dağılımından bahsediyor
# Price değişkeninin dağılımının değişkenliğinin nasıl oluştuğunu anlamamız gerekir

(sns
 .FacetGrid(df,
           hue = 'cut',
           height = 5,
           xlim = (0,10000))
 .map(sns.kdeplot, 'price', shade = True)
 .add_legend()
 );
 
 # price değişkeni ile cut değişkenini çaprazladık
 # price ı hangi sınıfın nasıl etkilediğini görebiliyoruz


# Bir boyut daha eklemek için
sns.catplot(x='cut', y = 'price', hue = 'color', kind = 'point', data = df)
# Fiyatın farklı seviyelerinde 2 kategorik değişkenin etkilerini gözlemleyebiliyoruz
# Price ın farklı fiyat seviyeleri söz konusu olduğunda cut ve color değişkenleri ayırt edici iki farklı değişken olabilir
# Model kurma aşaması için bu ayırt edicilik değerlidir
# E ve D gruplarının kendi içlerinde standart sapması daha düşüktür - aynı zamanda fiyat karşılığı da düşüktür



## BOXPLOT
# Süreli değişkenlerin dağılımını incelemek için kullanılır
import seaborn as sns
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()

df.shape

df.dtypes

df.describe()
df.describe().T

sns.boxplot(x = df['total_bill'])
sns.boxplot(x = df['total_bill'], orient='v')

# veri sıralandıktan sonra verinin dağılım yapısını bu şekilde görebiliyoruz
# Üst sınırdan sonraki değerlere aykırı gözlem muamelesi yapar

# hangi günler daha fazla kazanıyoruz
sns.boxplot(x = 'day', y = 'total_bill', data=df)
# haftasonları daha çok kazanıyoruz 

# yemek yemeye gelen kişilerin grup sayılarının kazanç ile orantısı
sns.boxplot(x = 'size', y = 'total_bill', data=df)

# Günün hangi saatinde çok kazanıyoruz
sns.boxplot(x = 'time', y = 'total_bill', data=df)

# sigara içen ile ödemeler
sns.boxplot(x = 'smoker', y = 'total_bill', data=df)

# 3. kırılımı hue fonksiyonu ile ekledi
sns.boxplot(x = 'day', y = 'total_bill', hue = 'sex', data=df)

# Sigara içmesi fark yaratmıyor denilebilir
sns.boxplot(x = 'smoker', y = 'tip', hue = 'sex', data=df)



# VIOLIN 
import seaborn as sns
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()

sns.catplot(y = "total_bill", kind="violin", data = df)
# Boxplot 'ın farklı bir versiyonu olarak düşünebilir
# Boxplot da köşeler keskindi ama burada dağılımın çeşitli köeşelerde nasıl şekillendiğini görebiliyoruz

sns.catplot(x = 'day', y='total_bill', kind='violin', data=df)

sns.catplot(x = 'day', y='total_bill', kind='violin', hue = 'sex', data=df)



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# çaprazlamalaraı farklı grafiklerde göstermek
g = sns.FacetGrid(df, row = "sex", col = "day", margin_titles = True)
bins = np.linspace(0,60,13)
g.map(plt.hist, "total_bill", color = "steelblue", bins = bins)
# FACETGRID ile grafikleri böldük




# KOLERASYON 
## BASIT SCATTERPLOT
import seaborn as sns
iris = sns.load_dataset("iris")
df = iris.copy()
df.head()

sns.pairplot(df)
# eksendeki elemanlar değişkenlerin histogramları
# kesişimlerdeki elemanlar o 2 değişken arasındaki ilişki
 
sns.pairplot(df, hue = "species")
# kırılım olarak bir kategorik değişken eklendi

sns.pairplot(df, hue = "species", markers = ["o","s","D"])
# Noktaların şekillerini farklılaştırmak

sns.pairplot(df, vars = ["sepal_width","sepal_length"])
# artış azalışı görmek için bir regresyon doğrusu eklemeliyiz

sns.pairplot(df, vars = ["sepal_width","sepal_length"],kind = "reg")

sns.pairplot(df, vars = ["sepal_width","sepal_length"], hue = "species",kind = "reg")
# iz gibi görünenler güven aralıkları - değişkenlik yorumu yapılabilir

g = sns.PairGrid(df, diag_sharey = False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_diag(sns.kdeplot, lw =3)

## HEATMAP
import seaborn as sns
flights = sns.load_dataset("flights")
df = flights.copy()
df.head()
df.shape

df = df.pivot("month", "year", "passengers")
df.head()
# eğer pattern detection veya yıl ve ay kırılımında mevsimsellik incelemesi yapmak istediğimizde bir ısı haritası kullanılabilir

sns.heatmap(df)
# en yüksek değerler açık değerler
# amaç sadece her yılın mart aylarını incelemek ise - 10 yıllık süreçte - uçuş sayısını incelemek istediğimde bu grafik çok kullanışlı

sns.heatmap(df, annot = True, fmt = 'd')

sns.heatmap(df, annot = True, fmt = 'd', linewidths = 0.5)

sns.heatmap(df,cbar = False)

'''
Çıktı Boyutu Değiştirmek için
import seaborn as sns
sns.set_context("poster")
sns.set(rc = {'figure.figsize' : (5,5)})
sns.set_style("whitegrid")
'''




# LINEGRAPH
import seaborn as sns
fmri = sns.load_dataset("fmri")
df = fmri.copy()
df.head()
df.shape

df["timepoint"].describe()
df["signal"].describe()

sns.jointplot(x="timepoint", y="signal", data =df, kind = "reg")
# zamana göre sinyalin durumu
# amaç buradaki pik durumunu anlamaya çalışmak
# her bir zaman noktasında 1den fazla sinyal var dolayısıyla bir groupby çekilebilir

df.groupby("timepoint")["signal"].count()
# her noktada eşit sayıda sinyal var mı?

# Her bir sinyalin eşsiz olmasını bekleriz
df.groupby("signal")["timepoint"].count()

# Sinyallerin eşsiz olduğunu gösterdik
df.groupby("signal").count()


df.groupby("timepoint").describe()


sns.lineplot(x= "timepoint", y = "signal", data = df)
# Çizgi ortalama değerlere karşılık gelir, boyamalar ise güven aralıklarını gösterir

sns.lineplot(x="timepoint", y = "signal", hue = "event", data = df)
sns.lineplot(x="timepoint", y = "signal", hue = "event", style = "event", data = df)
sns.lineplot(x="timepoint", y = "signal", 
             hue = "event", style = "event", 
             markers = True, dashes = False, data = df)

sns.lineplot(x="timepoint", y = "signal", hue = "region", 
             style = "event",data = df)





# ZAMAN SERİSİ GRAFİKLERİ
!pip install pandas-datareader
from pandas_datareader import data
apple = data.DataReader("AAPL", 'yahoo', 2014, 2018)
df = apple.copy()
df.head()
 
import matplotlib.pyplot as plt
import seaborn 
kapanis = df["Close"]

kapanis.plot()
# Burada bir index problemi var

kapanis.index = pd.DatetimeIndex(kapanis.index)

kapanis.plot()

# grafiğin boyutunu ve kalitesini yükseltme
%config InlineBeckend.figure_format = 'retina'
import seaborn as sns
sns.set_context("poster")
sns.set(rc={'figure.figsize': (16,9)})
sns.set_style("whitegrid")
kapanis.plot(style="-")

# hareketli ortalamalar yöntemi 
# hisse senedi fiyatlarının 3-5 günlük durumunu anlamaya çalışır

yedi = kapanis.rolling(window = 7). mean()
on = kapanis.rolling(window = 10). mean()
yirmi = kapanis.rolling(window = 20). mean()
otuz = kapanis.rolling(window = 30). mean()

yedi.head(10)
on.head(15)

kapanis.plot(style="-")
yedi.plot(style="--")
on.plot(style = "-.")
yirmi.plot(style = ":")
otuz.plot()
plt.legend(["input", "yedi", "on", "yirmi", "otuz"], loc = "upper left")

#zoom
plt.xlim("2017","2018")

# 7 günlük reaksiyonlar en erken tepki verir-trend ile ilgili bilgiyi buradan alabiliriz
