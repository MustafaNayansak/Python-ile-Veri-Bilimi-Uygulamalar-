# ************************** DERIN OĞRENME *********************
# ARTIFICIAL NEURAL NETWORK
# İnsan sinir hücrelerini araştırılıyor ve karar verme süreçlerinde bu hücrelerin etkili olduğu biliniyor
# Amaç insanın öğrenme ve karar verme yapısını bilgisayarlara programlama ve insanlar gibi düşünebilmelerini sağlamak 
# Sinir sisteminin 2 tane yapısı var;
# Nöron ve sinapsis adındaki iki yapı birbirine bağlanarak bir akım oluşturuyorlar
# Girdi Sistem ve Çıktı işlemleri var(ipo)
# (NÖRONA) Giriş Katmanı- Gizli Katman - Çıkış Katmanı adında 3 katman var
# NN'de bağımsız değişkenin standardize veya normalize(double) edilmiş olması gerekiyor
# Bazı sistemler - yöntemler veriyi kendi içine standardize ediyor 
# Girdi bir sinyal olarak geliyor - nörona kadar taşınırken üzerinde bir çarpan ile taşınabiliyor nöronda topolanıyor ve karar verme işlemi yapıyor7
# Aktivasyon Fonksiyonu ile nöronu aktive edilip edilmeyeceğine karar veriyor
# Girdilerin oluşturduğu bir yük var, aktivasyon fonksiyonuna geliyor, burada ateşlenip ateşlenmeyeceği kararı veriliyor
# Aktivasyon fonksiyonu olarak genelde Lojistik Regresyonda kullandığımız Sigmoid fonksiyonu kullanılıyor
# Aktivasyon fonksiyonu - eşik fonksiyonu veya treshold function da deniyor
# Aktivasyon fonksiyonu köşeli(step funciton) sigmoid gibi de düşünülebilir
# Aktivasyon fonksiyonu Hiperbolik Tanjant fonksiyonu da yine S şeklinde fakat -1 ile 1 arasında değerler alır
# Aktivasyon fonksiyonunun sonuca etkisi çok büyüktür, farklı aktivasyon fonksiyonlarını deneyerek sonuçları test edebiliriz
# Katman Kavramı(Layer) - giriş katmanı(Xiler) 
# Katman Kavramı(Layer) - Hidden layer girişten ve çıkıştan gizlidir. Farklı katmanlar veya nöron sayıları değişebilir
# Katman Kavramı(Layer) - çıkış katmanı(yi)
# Katman Kavramı(Layer) - girişler ağırlıklandırılıyor - aktivasyon fonksiyonuna aktarılıyor - sinyal ateşleyip ateşlemeyeceği belirleniyor - bu davranışlara göre kombinasyonlar belirlenir - ateşlenenler çıkış katmanı nöronuna aktarılıyor
# Katman Kavramı(Layer) - çıkış katmanında yükler yine toplanıyor ve burdaki farklı aktivasyon fonksiyonu ile bir çıktı elde edilebilir
# AND GATE(Ve Kapısı) - YY - Y, YD-Y, DD-D şeklindek kurulur(Threshold =1.5)
# Or Gate(Veya Kapısı) - YD - D, YY - Y, DD - D şeklinde kurulur(Threshold = 0.5)
# XOR Gate(ÖZel veya kapısı) - 00-0, 01-1, 10-1, 11-0 şeklinde oluşturabilir miyiz? Çözemeyiz! 
# AND ve OR gate doğrusal olarak ayrıştırılabilir - tek bir çizgi ile doğrusal olarak bölünemiyorsa doğrusal olarak ayrıştırılabilen problem değildir diyoruz 
# Buna çözüm olarak 1+1 nöron ile uzayı 2 çizgi ile bölebiliyoruz

# Öğrenmenin Parametreleri - Tek bir nörondan oluşuyorsa W1, W2 ve Threshold değişkenleridir
# Öğrenmenin Parametreleri - Perceptron Yöntemi - Bu yapay zeka bir cinsiyet tahmini yapıyor dicek
# Test ve Train setinde - train setinde algoritma sonucu ve girdileri biliyordu - perceptron'da aslında gerçek olan cinsiyet ile tahmin edilen cinsiyet arsındaki farkı alıp bu bilgiyi geri beslemek için kullanıyor
# Cinsiyet tahmini doğruysa hiçbir değişiklik olmuyor - cinsiyet tahmini ile gerçek arasında fark varsa bu bilgiyi neural networke geri yansıtarak yapay sinir ağındaki ağırlıkları değiştirmeye çalışıyor
# Amaç tahmin değerini gerçek değere yaklaştırmaya çalışıyoruz
# Geri besleme için kullanılan değer(1/2)*(gerçek-tahmin)^2 olarak hesaplanıyor 
# c değeri geri yansıyor ve weight değerlerini güncelliyoruz
# Learning Rate : c hatasının ne kadar geri yansıtılacağı bir çarpan ile çarpılabiliyor(sistemin ani tepkiler vermesini önlemek için, öğrendiklerini unutabilir)
# Learning rate : c değerinin sisteme nasıl yansıtılacağını hesaplar

# İlk yapılması gereken verileri normalize etmek(0 ile 1 arasında oturtmak)
# Öncelikle weightlere ve eşit değerine 1 veriyoruz. Makine kendisi öğreneceği için değer verme işlemi kritik değil
# Learning Rate belirlenir ve geri çarpan olarak çarpılıyor ve ceza olarak geri dönderiliyor
# Geri dönen değer - olarak geliyor
# Learning rate giderek düşen bir değer

# Gradyan Alçalış : 2 giriş 3 gizli katman olduğu yapay zekada 6 tane bağlantı vardır
# Gradyan Alçalış - Burada güncellenecek 6 adet değer vardır anlamına geliyor
# Gradyan AlçalışBüyük öğrenme ve küçük öğrenme oranı adında iki adet öğrenme oranı(atlamalar) bulunuyor
# Gradyan Alçalış - Öğrenme oranı doğru yere gidip gitmediğimizi de gösteren bir orandır

# Stokastik Gradyan Alçalış
# Her bir verinin sonunda alçaltma veya değiştirme gibi kararlar alıyorsak buna gradient descendent diyoruz
# Stokastik : Tüm sistemi görmeden örnek yapıya bakarak yaptığımız iş 
# Batch(yığın) : Tüm veriyi okuduktan sonra sisteme geri besleme yapılıyor
# Mini Batch : Stokastik ile Batch arasında bir yaklaşım. Her 100 örnekte bir karar ver gibi bir yaklaşım 
# Amaç optimum noktayı bulmak

# Backpropogation (Geri Yayılım) - çıkıştan girişe doğru olan bir yayılım türüdür
# Backpropogation (Geri Yayılım) - 1 bütün ağı rastgele sayılar ile ilklendiriyoruz
# Backpropogation (Geri Yayılım) - 2 veri kümseinden ilk satır giriş katmanına verilir(değişkenler(kaç tane nöron olacağını öznitelik sayısı belirler))
# Backpropogation (Geri Yayılım) - neural network ne kadar fazla özniteliğe sahip olsa da bu özelliklere bağışık bir yapıya sahiptir(ne kadar nöron verdiğinizle de ilgili bir durum)
# Backpropogation (Geri Yayılım) - 3 ileri yönlü yayılım yapıyor. sinir ağır istenen sonucu verene kadar güncelleniyor
# Backpropogation (Geri Yayılım) - 4 Gerçek ve çıktı arasındaki fark(hata) hesaplanır
# Backpropogation (Geri Yayılım) - 5 Geri yayılıma başlıyoruz ve Her sinapsis üzerindeki hata güncellenir. Forward Prep. ile hangi sinapsisin hatadan daha fazla sorumlu olduğunu verebiliyor
# Backpropogation (Geri Yayılım) - 6- 1-5 arası adımlar istenen sonucu elde edilene kadar tekrar edilir.
# Backpropogation (Geri Yayılım) - 7 Bütün eğitim kümesi çalıştırıldıktan sonra bu bir tur(epoch) oluyor ve ondan sonra aynı veri kümesi üzerinde çalıştırılıyor

# Öğrenme oranı - kaç tur atılacağı(epoch) değerlerini sabitlemeliyiz. Öğrenirken her turda aynı verilerin üzerinden geçiyor bunun belirli bir zamandan sonra sisteme olumlu etkisinin olup olmadığını bilemiyoruz. 
# Düşük verirsek öğrenme süreci tamamlanmaz yüksek verirsek tekrar tekrar aynı ypaay sinir ağı üzerinden geliştirme yapar anlamsız
