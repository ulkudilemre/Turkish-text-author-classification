Mevcut projede daha önceden, doğal dil işleme kütüphanesi olarak kullanılan NLTK kütüphanesi yerine Türkçe metinlerde morfolojik analiz yapmamıza olanak sağlayan Zemberek
kütüphanesine geçilmeye karar verilmiştir. Zemberek Java tabanlı ve proje Python tabanlı olduğundan dolayı Zemberek, Jpype kütüphanesinin yardımı ile projeye entegre 
edilmiştir. 

Daha önceden veri kazıma ile elde edilen verilerde 204 yazara ait 2415 köşe yazısı bulunmaktadır. Veriler dengesizdir: 15’ten az köşe yazısı bulanan yazar sayısı 95, bu 
yazarların toplam köşe yazısı sayısı 328’tir. 15’ten az köşe yazısı bulunan yazarların sayısı fazla, fakat toplam köşe yazısı sayısı veri setini küçültmediği için bu 
veriler veri setinden çıkartılmıştır.

Yazı formatındaki verilerin makine öğrenmesine hazırlanması için uygulanan işlemler şunlardır; metni küçük harfe çevirme, kelime belirteçlerine ayırma, kelimelerin 
eklerini atarak köklerini çıkarma, stopword kelimelerin atılması, cümle belirteçlerine ayırma, kök uzunluk dağılımlarının çıkartılması, kelime bazında cümle uzunluk 
dağılımlarının çıkartılması, kelime zenginlik oranlarının çıkartılması, ortalama kök uzunluğu, kelime olarak ortalama cümle uzunluğu, toplam noktalama işareti sayısının 
çıkartılması, toplam kullanılan stopword sayısının çıkartılması, tamamen büyük harfte yazılmış kelime sayısının çıkartılmasıdır. Yazarlar pozitif tam sayılar olarak 
kodlanmıştır.

Ön işlemden geçirilerek kökleri çıkartılan her metnin TF-IDF, kelime torbası vektörleri, kelime, cümle dağılımları ve diğer özniteliklerin vektörleri çıkartılmış, 
tüm vektörler kendi aralarında min-max normalizasyon yöntemi kullanılarak normalleştirme yapılmıştır. 

Verilerin %77’si eğitim (1398 köşe yazısı), %33’ü test (689 köşe yazısı) olmak üzere ikiye ayrılmıştır. Scikit-learn kütüphanesindeki SVM modeli varsayılan parametrelerde 
eğitilerek özniteliklerin farklı kombinasyonları ile alınan doğruluk puanları karşılaştırılmıştır. En yüksek %57,4 doğruluk oranına ulaşılmıştır. SVM ile en iyi doğruluk
oranına ulaşılan öznitelikler ile, yine Scikit-learn kütüphanesinde bulunan MLP Classifier (Multi-Layer Perceptron, Çok Katmanlı Algılayıcılar), sinir ağı modeli 
varsayılan parametrelerinde eğitilerek %77 doğruluk oranına ulaşılmıştır. 

Ardından eğitim ve test verilerinde, veri dağılıma bağlı sapmalar ve hatalardan kaçınmak, ayrıca SVM modelinde verilere en uygun parametreleri belirlemek için 
Scikit-learn kütüphanesinde bulunan RandomizedSearchCV fonksiyonu kullanılmıştır. Sürekli bir olasılıksal dağılım oluşturan SciPy kütüphanesinde bulunan log-uniform 
fonksiyonu ile C değerleri 1-1000 arasında, gamma değeri 0.0001-0.01 arasında seçilmiş ve toplam 30 iterasyonda, 10 kat çapraz doğrulama ile hiper-parametreler 
aranmıştır. En yüksek doğruluk puanı %85 ile linear çekirdekte C = 216 değeridir. Linear çekirdekte gamma değeri bulunmamaktadır.
