# Numpy (Numerical Python)
'''
# Döngülerden Vektörel Operasyonlara
# Array ve matrisler üzerinde yüksek performanslı çalışma imkanı sağlar
# Python'ın analitik dünyasının zeminidir. Ana kütüphane olarak nitelendirilebilir
'''

a = [1,2,3,4]
b = [2,3,4,5]
ab = []

for i in range(0, len(a)):
    ab.append(a[i]*b[i])
    
ab


import numpy as np
a = np.array([1,2,3,4])
b = np.array([2,3,4,5])
a*b

'''
* ?np ile başına ? koyarak dokümantasyona ulaşabiliriz
* Python C dili ile yazılmıştır. 
'''

# Numpy Arrayleri Oluşturma

## Listelerden Arrey Oluşturmak
import numpy as np
np.array([12,33,4,5])
a=np.array([12,33,4,5])
a
type(a)

np.array([3.14,4,5,1.2])
# Array oluşturduğumda köşeli parantez kullan


# Sıfırlardan oluşan bir seri
np.zeros(10, dtype = int)


# Satır ve sütunları belirtilen 1'lerden oluşan matris
np.ones((2,3))


np.full((2,3),9)


np.arange(0,10,2)


np.linspace(0,1,30)


np.random.normal(0,1,(3,4))


np.random.randint(0,10,(2,2))

np.eye(3)



## Numpy Biçimlendirme 
'''
ndim: Boyut Sayısı
shape: Boyut Bilgisi
size: Toplam Eleman Sayısı
dtype: Array Veri Tipi

'''

import numpy as np
a = np.random.randint(10, size = 10)
a.ndim
a.shape
a.size
a.dtype


a = np.random.randint(10, size = (10,5))
a.ndim
a.shape
a.size
a.dtype



# Reshaping 
np.arange(1,10).reshape((3,3))

a = np.array([1,2,3])
a

# Elimizdeki arrayi matrixe çevirme işlemi
b = a.reshape((1,3))
b

b.ndim

a[np.newaxis,:]
a[:,np.newaxis]



## Array Birleştirme İşlemi
x = np.array([1,2,3])
y = np.array([4,5,6])
x
y
np.concatenate([x,y])

z = [1,2,3]
np.concatenate([x,y,z])


# İki Boyutlu
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
a

np.concatenate([a,a])
np.concatenate([a,a],axis=1)

# Farklı boyutlu 
a = np.array([1,2,3])
b = np.array([[9,8,7],[6,5,4]])
a
b
np.vstack([a,b])

a = np.array([[99],[99]])
a

np.hstack([a,b])


## Splitting(Ayırma İşlemi)
x = [1,2,3,9,99,3,2,1]
x

np.split(x,[3,5])

a,b,c = np.split(x,[3,5])

m = np.arange(16).reshape((4,4))
m


# Dikey
np.vsplit(m,[2])


ust, alt = np.vsplit(m,[2])
ust
alt

# Yatay
np.hsplit(m, [2])

sag, sol = np.hsplit(m, [2])

v = np.array([2,3,1,5,3,2])
v
np.sort(v)
v
# Bu şekilde v değişkenini güncelleyebilirsin(sort)
v.sort()
v = np.array([2,3,1,5,3,2])
np.sort(v)


# Yaptığımız sıralama işlemi sonrasında index değişklkleri
# Sıralama indexleri
i = np.argsort(v)
i


# Numpy Eleman İşlemleri
import numpy as np
a = np.random.randint(10,size = 10)
a 

a[0]

a[-2]

a[0] = 1
a

a = np.random.randint(10,size = (3,5))
a 

a[0,0]

a[0,0] = 2
a

a[0,0] = 2.2
a


## Slicing ile Array Alt Kümelerine Erişmek
a = np.arange(20,30)
a
a[0:3]
a[3:]
a[::2]
a[1::2]
a[2::2]
a[1::3]

a = np.random.randint(10, size = (5,5))
a
a[:,0]
a[:,1]
a[0,:]
a[0]
a[:2,:3]
a[0:2,0:3]
a[::,:2]
a[1:3,0:2]


# Array Alt Kümelerini Bağımsızlaştırmak
a = np.random.randint(10, size = (5,5))
a
alt_a = a[0:3,0:2]
alt_a

alt_a[0,0] = 9999
alt_a[1,1] = 9999
alt_a 

a
#alt_a da yapılan değişiklikler a yı da etkiledi

a = np.random.randint(10, size = (5,5))
a

alt_b = a[0:3,0:2].copy()
alt_b

alt_b[0,0] = 9999
alt_b[1,1] = 9999

alt_b


## Fancy Index ile Eleman Islemleri
v = np.arange(0,30,3)
v
v[1]
v[3]
[v[1],v[3]]

al_getir = [1,3,5]
v[al_getir]

m = np.arange(9).reshape((3,3))
m

satir = np.array([0,1])
sutun = np.array([1,2])

m[satir,sutun]
# Kesişimleri

m[0,[1,2]]
m[0:,[1,2]]

v = np.arange(10)
v

index = np.array([0,1,2])
index

v[index] = 99
v

v[[0,1]] = [4,6]
v


## Koşullu Eleman İşlemleri
v = np.array([1,2,3,4,5])
v>3
v <= 3
v == 3
v != 3 
(2*v)
v**2

#ufunc
np.equal(3,v)

np.equal([0,1,3],np.arange(3))

v = np.random.randint(0,10,(3,3))

v > 5

np.sum(v>5)

np.sum((v>3) | (v<7))

np.sum(v > 4, axis = 1)

# Verideki tüm elemanlar 4ten büyük mü değil mi?
np.all(v>4)

np.any(v>4)

# axis = 0 sütun 1 satır bazında işlem yapar
np.any(v>4, axis = 1)

v = np.array([1,2,3,4,5])

v[v>3]

v[(v>1) & (v<5)]



# Numpy Hesaplamalı İşlemler
a = np.array([0,1,2,3,4])
a

np.add(a,2)

np.subtract(a,1)

np.divide(a,3)

a = np.arange(1,6)
a

np.add.reduce(a)

np.add.accumulate(a)

a = np.random.normal(0,1,30)
a

np.mean(a)

np.std(a)

np.median(a)

np.min(a)

a = np.random.normal(0,1,(3,3))
a

# Satır Bazında toplamlar
a.sum(axis=1)



# Farklı Boyutlu Arrayler ile Çalışmak(Broadcasting)
# Broadcasting yaymak anlamına gelmektedir
import numpy as np
a = np.array([1,2,3])
b = np.array([1,2,3])
a+b

m = np.ones((3,3))
m

a+m

a = np.arange(3)
a

b = np.arange(3)[:,np.newaxis]
b

a+b



# Hello and Goodbye Numpy!
isim = ['ali', 'veli','isik']
yas = [25,22,19]
boy = [168,159,172]

import numpy as np
data = np.zeros(3,dtype = {'names':('isim','yas','boy'),
                             'formats':('U10','i4','f8')})
data

data ['isim'] = isim
data ['yas'] = yas
data ['boy'] = boy
data

data[0]
data[data['yas'] < 25]['isim']

