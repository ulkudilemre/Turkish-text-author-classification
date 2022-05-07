# English

In the current project, it has been decided to switch to the ***Zemberek*** library, which allows us to perform morphological analysis in Turkish texts, instead of the NLTK library, which was previously used as a natural language processing library. Since Zemberek is Java-based and the project is Python-based, Zemberek has been integrated into the project with the help of the Jpype library.

There are ***2415 texts*** belonging to ***204 authors*** in the data obtained by previous data scraping. The data is uneven: the number of authors with less than 15 texts is 95, and the total number of texts for these authors is 328. The number of authors with less than 15 columns is high, but these data were excluded from the data set because the total number of texts was not large.

The processes applied to prepare the data in text format for machine learning are as follows; 
1. converting the text to lowercase
2. separating the word markers 
3. removing the roots of the words by removing the suffixes
4. removing the stopword words
5. separating the sentence markers
6. extracting the root length distributions
7. extracting the sentence length distributions on the basis of words
8. extracting the word richness ratios
9. average root length
10. average sentence length as a word
11. subtracting the total number of punctuation marks
12. subtracting the total number of stopwords used
13. subtracting the number of words written in all capital letters
14. Authors were coded as positive integers.

The vectors of ***TF-IDF, word bag vectors, word, sentence distributions*** and other features of each text whose roots were extracted by pre-processing were extracted, and all vectors were normalized among themselves using the ***min-max normalization*** method.

77% of the data was divided into two as education (1398 columns), 33% as test (689 columns). The SVM model in the Scikit-learn library was trained on the default parameters and the accuracy scores obtained with different combinations of the features were compared. The highest accuracy rate of 57.4% was achieved. With the features that have the best accuracy rate with SVM, the ***MLP Classifier*** (Multi-Layer Perceptron, Multi-Layer Perceptron), which is also in the ***Scikit-learn*** library, was trained on the neural network model default parameters, and an accuracy rate of 77% was achieved.

Then, the ***RandomizedSearchCV*** function in the **Scikit-learn** library was used in the training and test data to avoid deviations and errors due to data distribution, and to determine the most suitable parameters for the data in the **SVM** model. With the log-uniform function in the SciPy library, which creates a continuous probabilistic distribution, the C values were selected between 1-1000, the gamma value was between 0.0001-0.01, and hyper-parameters were searched with 10-fold cross-validation in a total of 30 iterations. The highest accuracy score is ***C = 216 in the linear kernel with 85%***. There is no gamma value in the linear kernel.

# Türkçe

Mevcut projede daha önceden, doğal dil işleme kütüphanesi olarak kullanılan NLTK kütüphanesi yerine Türkçe metinlerde morfolojik analiz yapmamıza olanak sağlayan ***Zemberek*** kütüphanesine geçilmeye karar verilmiştir. Zemberek Java tabanlı ve proje Python tabanlı olduğundan dolayı Zemberek, Jpype kütüphanesinin yardımı ile projeye entegre 
edilmiştir. 

Daha önceden veri kazıma ile elde edilen verilerde ***204 yazara*** ait ***2415 köşe yazısı*** bulunmaktadır. Veriler dengesizdir: 15’ten az köşe yazısı bulanan yazar sayısı 95, bu 
yazarların toplam köşe yazısı sayısı 328’tir. 15’ten az köşe yazısı bulunan yazarların sayısı fazla, fakat toplam köşe yazısı sayısı veri setini küçültmediği için bu 
veriler veri setinden çıkartılmıştır.

Yazı formatındaki verilerin makine öğrenmesine hazırlanması için uygulanan işlemler şunlardır;
1. metni küçük harfe çevirme
2. kelime belirteçlerine ayırma
3. kelimelerin eklerini atarak köklerini çıkarma
4. stopword kelimelerin atılması
5. cümle belirteçlerine ayırma
6. kök uzunluk dağılımlarının çıkartılması
7. kelime bazında cümle uzunluk dağılımlarının çıkartılması
8. kelime zenginlik oranlarının çıkartılması
9. ortalama kök uzunluğu
10. kelime olarak ortalama cümle uzunluğu
11. toplam noktalama işareti sayısının çıkartılması
12. toplam kullanılan stopword sayısının çıkartılması
13. tamamen büyük harfte yazılmış kelime sayısının çıkartılmasıdır
14. Yazarlar pozitif tam sayılar olarak kodlanmıştır.

Ön işlemden geçirilerek kökleri çıkartılan her metnin ***TF-IDF***, ***kelime torbası vektörleri***, ***kelime, cümle dağılımları*** ve diğer özniteliklerin vektörleri çıkartılmış, tüm vektörler kendi aralarında **min-max normalizasyon** yöntemi kullanılarak normalleştirme yapılmıştır. 

Verilerin %77’si eğitim (1398 köşe yazısı), %33’ü test (689 köşe yazısı) olmak üzere ikiye ayrılmıştır. **Scikit-learn** kütüphanesindeki ***SVM*** modeli varsayılan parametrelerde 
eğitilerek özniteliklerin farklı kombinasyonları ile alınan doğruluk puanları karşılaştırılmıştır. En yüksek %57,4 doğruluk oranına ulaşılmıştır. SVM ile en iyi doğruluk
oranına ulaşılan öznitelikler ile, yine Scikit-learn kütüphanesinde bulunan ***MLP Classifier*** (Multi-Layer Perceptron, Çok Katmanlı Algılayıcılar), sinir ağı modeli 
varsayılan parametrelerinde eğitilerek %77 doğruluk oranına ulaşılmıştır. 

Ardından eğitim ve test verilerinde, veri dağılıma bağlı sapmalar ve hatalardan kaçınmak, ayrıca SVM modelinde verilere en uygun parametreleri belirlemek için 
Scikit-learn kütüphanesinde bulunan ***RandomizedSearchCV*** fonksiyonu kullanılmıştır. Sürekli bir olasılıksal dağılım oluşturan SciPy kütüphanesinde bulunan log-uniform 
fonksiyonu ile C değerleri 1-1000 arasında, gamma değeri 0.0001-0.01 arasında seçilmiş ve toplam 30 iterasyonda, 10 kat çapraz doğrulama ile hiper-parametreler 
aranmıştır. En yüksek doğruluk puanı ***%85 ile linear çekirdekte C = 216 değeridir***. Linear çekirdekte gamma değeri bulunmamaktadır.
