# Pandas 
'''
* Dataframeler üzerinde hızlı ve etkili çalışabilme imkanı sağlamıştır
* Bir çok farklı veri tipini okuma ve yazma imkanı sağlar
* Numpy'ın özelliklerini kullanan ve bu özellikleri geliştiren bir kütüphane.
'''

import numpy as np
import pandas as pd

seri = pd.Series([1,2,3,4])
type(seri)
seri.dtype
seri.empty
seri.ndim
seri.values
seri.head(2)
seri[0:3]

a = np.array([1,2,3,4,4,4,444])
seri = pd.Series(a)
seri

pd.Series([1.1,4,3,3], index=[1,3,4,5])

seri = pd.Series([1.1,4,3,3], index=['a','b','c','d'])
seri['a']

sozluk = { " reg " : 10, " loj " : 11, " cart " : 12}
sozluk

seri = pd.Series(sozluk)
seri[" reg "]

seri[" reg " : " cart "]
pd.concat([seri,seri])
seri

"knn" in seri
"reg" in seri
seri[" reg "]

# Fancy indexte iki eleman kullanılır 
seri[[ ' reg ']]
seri[' loj ':' cart ']


# Koşullu Eleman Seçme İşlemi
seri = pd.Series([121,200,150,99], index=["reg", "loj", "cart", "rf"])
seri
seri[(seri>125)&(seri<200)]

data = pd.Series(['a','b','c'], index=[1,3,5])
data

# data[0] dediğimizde oluşturduğumuz indexleme yapısı biz tarafından belirlendi ve 1den başladı
# indexin oluşturulduğu şekliyle çağırmamız gerekmektedir.
data[3]

data[1:3]
# a değerinin burada olması gerekir

data[0:3]
# slicing işlemlerinde net indexe gitmemiz gerekmektedir


# LOC : Label Based Indexing
# data.loc[2]
# İndeksleri tanımlandığı şekilde yakalamak için loc kullanılır
data.loc[0:2]
# Bu aralıkta 3 indeksini barındırmadığımız için 3. değeri getirmedi
# Aralık verdiysek bile tanım aralığı içerisinde yer alıyorsa getirir

data.loc[0:4]

# ILOC : Positional Indexing
# Indeksi sıfırlayarak yakalamak
data.iloc[2]




# Pandas DataFrame Ozellikleri&Oluşturma
l = [1,2,666,66,666,3333]
l

df = pd.DataFrame(l,columns = ['degisken_ismi'])
df.axes
df.shape
df.ndim
df.values
df.tail(2)

a = np.array([1,2,3,544])
pd.DataFrame(a, columns = ['degisken_ismi'])

m = np.arange(1,10).reshape((3,3))

df = pd.DataFrame(m, columns = ['var1', 'var2', 'var3'])
df

df.columns = ('deg1', 'deg2', 'deg3')
df

pd.DataFrame(m, columns = ['var1', 'var2', 'var3'], index=['a','b','c'])

pd.Series([12,3,4,4])

pd.DataFrame(pd.Series([12,3,4,4]), columns = ['degisken'])

bir = pd.Series([12,3,4,4])
iki = pd.Series([12,3,4,4])

pd.DataFrame({'degisken1': bir, 'degisken2': iki })

sozluk = {"reg": {"RMSE":10,
                 "MSE":11,
                 "SSE":90},

         "loj": {"RMSE":89,
                 "MSE":12,
                 "SSE":45},
         
         "cart": {"RMSE":45,
                 "MSE":22,
                 "SSE":11}}
pd.DataFrame(sozluk)




## Pandas DF-Eleman İşlemleri
s1 = np.random.randint(10, size = 5)
s2 = np.random.randint(10, size = 5)
s3 = np.random.randint(10, size = 5)

df = pd.DataFrame({"var1" : s1, "var2" : s2, "var3":s3})
df

df[0:1]

df.index = ['a', 'b', 'c','d','e']
df

df['c':'e']

df.drop('a', axis = 0)

df
#Yaptığımız işlem df e yansımadı bunun için inplace argümanı kullanılır

df.drop('a', axis =0, inplace=True)
df

l = ["b","c"]
l

df.drop(l,axis=0)


# Değişkenlere Erişmek ve Seçim İşlemleri
"var1" in df
l = ['var1', 'var2', 'var4']
for i in l:
    print(i in df)
df['var1'] is df['var2']
df['var1'] is df['var1']

# Attribute tibi 
df.var2

df[['var1','var2']]

l = ['var1', 'var2']
df[l]

df['var4'] = df['var1']/df['var2']
df

df.drop('var4', axis = 1)
#Kalıcı değil
df

df.drop('var4', axis = 1, inplace=True)
#Kalıcı



## Birleştirme İşlemleri 
s1 = np.random.randint(10, size = 5)
s2 = np.random.randint(10, size = 5)
s3 = np.random.randint(10, size = 5)

df1 = pd.DataFrame({"var1" : s1, "var2" : s2, "var3":s3})
df1

df2 = df1 + 99
df2

pd.concat([df1,df2])
#indeksleme sıkıntısı var

pd.concat([df1,df2],axis = 1)

pd.concat([df1,df2], ignore_index = True)

df2.columns = ["var1", "var2","deg3"]
df2

pd.concat([df1,df2])
# Değerler yer almadığında NAN olarak döndürdü

pd.concat([df1,df2], join = 'inner')
# SAdece kesişen değerleri birleştirdik

pd.concat([df1,df2], join_axes = [df1.columns])
# 1. df i referans alarak değerleri birleştirdi

import numpy as np
import pandas as pd
df1 = pd.DataFrame({'calisanlar':['Ali','Veli','Ayşe','Fatma'],
                   'grup': ['Muhasebe', 'Muhendislik', 'Muhendislik', 'İK']})

df1

df2 = pd.DataFrame({'calisanlar':['Ali','Veli','Ayşe','Fatma'],
                   'ilk_giris': ['2010', '2009', '2014', '2019']})
df2




# Merge ile Birleştirme
pd.merge(df1,df2)
# one to one birleştirme
df3 = pd.merge(df1,df2)
# many to one
df

df4 = pd.DataFrame({'grup': ['Muhasebe', 'Muhendislik', 'İK'],
                   'mudur': ['Caner', 'Mustafa', 'Berkcan']})
df4

pd.merge(df3, df4)
# many to one

df5 = pd.DataFrame({'grup':['Muhasebe', 'Muhasebe','Muhendislik', 'Muhendislik',
                            'İK', 'İK'],
                   'yetenekler': ['matematik', 'excel', 'kodlama', 'linux',
                                 'excel', 'yonetim']})
df5

# many to many
pd.merge(df1,df5)
# Alinin sahip olduğu 2 yeteneği göstermek için aliyi çokladık


df3 = pd.DataFrame({'name': ['ali', 'veli', 'ayşe', 'fatma'],
                   'maas': [70000,80000,120000,90000]})
df3
df1
# Bir yerde name diğerinde çalışanlar şeklinde yazılmış

pd.merge(df1,df3)
pd.merge(df1,df3,left_on = 'calisanlar',right_on='name')
pd.merge(df1,df3,left_on='calisanlar', right_on='name').drop('name',axis=1)

# Veri setlerini değil de indexlerini birleştirmek istersek
df1a = df1.set_index('calisanlar')
df1a.shape

df2a = df2a = df1.set_index('calisanlar')
df2a

pd.merge(df1a,df2a, left_index=True,right_index=True)

df1a



dfa = pd.DataFrame({'calisanlar':['Ali', 'Veli', 'Ayse', 'Fatma'],
                    'siralama':[1,2,3,4]})
dfb = pd.DataFrame({'calisanlar':['Ali', 'Veli', 'Ayse', 'Fatma'],
                    'siralama':[2,1,4,3]})
             
pd.merge(dfa, dfb, on='calisanlar')
# a dan geleni x b den geleni y olarak birleştirdi
pd.merge(dfa, dfb, on='calisanlar',suffixes = ["_MAAS_","_DENEYİM_"])





# Aggregation ve Gruplama işlemleri
import seaborn as sns
df = sns.load_dataset('planets')
df
df.shape
df.count()
df['mass'].count()

df.describe()
df.describe().T

df['mass'].describe()

df.mean()


df.dropna().describe().T



# Gruplama İşlemi
a = pd.DataFrame({'gruplar': ['A','B', 'C','A','B','C'],
                   'veri': [10,11,45,33,77,55]}, 
             columns=['gruplar','veri'])
a

a.groupby('gruplar')
a.groupby('gruplar').sum()


df.head()
df.groupby('method')['orbital_period'].sum()
df.groupby('method')['orbital_period'].describe()




# AGGREGATE, FILTER, TRANSFORM, APPLY
df = pd.DataFrame({'gruplar':['A','B', 'C','A','B','C'],
                   'degisken1': [10,23,33,22,11,99],
                   'degisken2': [100,233,221,545,332,555]},
columns=['gruplar','degisken1','degisken2'])


df.groupby('gruplar').aggregate(['min'], np.median,max)
df.groupby('gruplar').aggregate({'degisken1': 'min','degisken2': 'max'})


# FILTER
def filter_func(x):
    return x['degisken1'].std()>9
df.groupby('gruplar').filter(filter_func)


# TRANSFORM
df.groupby('gruplar').transform(lambda x: (x-x.mean())/x.std())

# APPLY 
df.groupby('gruplar').apply(np.sum)
df.groupby('gruplar').apply(np.mean)

df.groupby(df['gruplar']).sum()

L = [0,1,0,1,2,0]
df.groupby(L).sum()
