import jpype
from jpype import JClass
import pandas as pd
import numpy as np
import string

# Zembrek'i çalıştıralım
jar = r"C:/Users/Dilemre/Documents/GitHub/bitirme/zemberek-full.jar" # Zemberek'in yolu
jvmpath = r"C:/Program Files/Java/jdk-17.0.2/bin/server/jvm.dll" # JVM'nin yolu
if not jpype.isJVMStarted():
    jpype.startJVM(jvmpath=jvmpath, classpath=jar)
    
try:
    TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
    TurkishSentenceExtractor = JClass('zemberek.tokenization.TurkishSentenceExtractor')
except:
    print("wrong path")
    
morphology = TurkishMorphology.createWithDefaults()
extractor = TurkishSentenceExtractor.DEFAULT

###

stopwords = [x.strip() for x in open('stop-words.txt','r', encoding="UTF8").read().split('\n')]

def dataCleaning(text):
    """Satır boşluklarını kaldır, metini küçük harfe çevir, noktalama işaretlerini kaldır"""
    text = text.replace("\n", " ")
    text = text.lower()
    text = "".join([i for i in text if (i.isalnum() or i == " ")])
    return " ".join(text.split())


def removeStopwords(text):
    """Zemberek'ten aldığımız stopword kelimelerini kaldırır"""
    for word in text:
        if word in stopwords or word in string.whitespace:
            text.remove(word)
    return text


def wordTokenize(text):
    """Önişlenmiş metini kelimelere ayırır ve stopword'leri kaldırır"""
    text = text.split(" ")
    text = removeStopwords(text)
    return text


def sentTokenize(text):
    """Önişlenmemiş metini cümlelerine ayırır, bunişlem sırasında önişleme yapar ve stopword'leri kaldırır """
    sent_list = []
    text = text.replace("\"", "")
    results: TurkishSentenceExtractor = extractor.fromDocument(jpype.JString(text))
    for result in results:
        result = dataCleaning(str(result))
        result = removeStopwords(result.split(" "))
        if len(result) == 0:
            continue
        else:
            sent_list.append(" ".join(result))         
    return sent_list

#### Burada UNK kelimeleri UNK olarak alıcak şekilde düzenlenmeli - hatta unk kelime sayısı da nitelik olarak eklenmeli
def lemmas(word_list):
    """Kelime token'larından kök tokenları oluşturur"""
    lemma = []
    for word in word_list:
        result = str(morphology.analyzeAndDisambiguate(word).bestAnalysis()[0].getLemmas()[0])
        if result == "UNK":
            lemma.append(result)
        else:
            lemma.append(result)
    return lemma


def wtDist(wt):
    """Kelimelerin dağılımları"""
    wt_dist = dict()
    wt_dist.fromkeys(set(wt))
    for i in set(wt):
        wt_dist[i] = wt.count(i)
    return wt_dist


def wtLenDist(wt):
    """Kelimelerin harf olarak uzunluk dağılımlarını çıkarır"""
    wt_len = [len(str(word)) for word in wt]
    wt_len_dist = dict()
    wt_len_dist.fromkeys(range(1, 29))
    for i in range(0, 29):
        wt_len_dist[i] = wt_len.count(i)
    return wt_len_dist


def stLenDist(st):
    "Cümlelerin kelime olarak uzunluk dağılımlarını çıkarır"
    st_len = [len(wordTokenize(sent)) for sent in st]
    st_len_dist = dict()
    st_len_dist.fromkeys(range(1, 29))
    for i in range(0, 29):
        st_len_dist[i] = st_len.count(i)
    return st_len_dist


def typeTokenRatio(wt):
    """Kelimenin toplam kelime sayısına oranı"""
    return len(set(wt))/len((wt))

def avgWtLen(wt):
    return sum(len(word) for word in wt)/len(wt)

def avgStLen(st):
    return sum(len(wordTokenize(sent)) for sent in st)/len(st)

def puncNum(txt):
    return len([x for x in txt if x in string.punctuation])

def numStopwords(clean_text):
    return len([w for w in clean_text.split() if w in stopwords])

def numUpper(raw_text):
    p = string.punctuation + "’" + "“" + "”"
    raw_text = ''.join(' ' if c in p else c for c in raw_text)
    return len([w for w in raw_text.split() if str(w).isupper()])

###

import pickle

with open('./model/mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

with open('./model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('./model/count_vectorize.pkl', 'rb') as f:
    count_vectorize = pickle.load(f)
    
with open('./model/tfidf_vector.pkl', 'rb') as f:
    tfidf_vector = pickle.load(f)
    
with open('./model/dict_vector.pkl', 'rb') as f:
    dict_vector = pickle.load(f)
    
with open('./model/SVM.pkl', 'rb') as f:
    svm = pickle.load(f)
    
###

def processing(text):
    
    data = pd.DataFrame(columns = ["text"])
    data = data.append({"text" : text}, ignore_index=True)
    
    data['clean_text'] = data['text'].apply(lambda x : dataCleaning(x))
    data['word_token'] = data['clean_text'].apply(lambda x : wordTokenize(x))
    data['sent_token'] = data['text'].apply(lambda x : sentTokenize(x))
    data['lemma_token'] = data['word_token'].apply(lambda x : lemmas(x))
    data['ltDist'] = data['lemma_token'].apply(lambda x : wtDist(x))
    data['wtLenDist'] = data['word_token'].apply(lambda x : wtLenDist(x))
    data['stLenDist'] = data['sent_token'].apply(lambda x : stLenDist(x))
    data['ttr'] = data['word_token'].apply(lambda x : typeTokenRatio(x))
    data['lttr'] = data['lemma_token'].apply(lambda x : typeTokenRatio(x))
    data['avgWtLen'] = data['word_token'].apply(lambda x : avgWtLen(x))
    data['avgStLen'] = data['sent_token'].apply(lambda x : avgStLen(x))
    data['puncNum'] = data['text'].apply(lambda x : puncNum(x))
    data['numStopwords'] = data['clean_text'].apply(lambda x : numStopwords(x))
    data['numUpper'] = data['text'].apply(lambda x : numUpper(x))
    
    
    # Vectorizing
    normalizedLabels = scaler.transform(data.loc[:, 'lttr':'numUpper'])

    x = np.array(data.lemma_token)
    for i in range(0, len(data.lemma_token)):
        x[i] = " ".join(data.lemma_token[i])
        
    sparce_matrix = count_vectorize.transform(x).toarray()
    sparce_matrix = (sparce_matrix - sparce_matrix.min())/(sparce_matrix.max() - sparce_matrix.min())
    
    tfidf_matrix = tfidf_vector.transform(x).toarray()
    
    wtLenDist_matrix = dict_vector.transform(data.wtLenDist).toarray()
    stLenDist_matrix = dict_vector.transform(data.stLenDist).toarray()
    wtLenDist_matrix = (wtLenDist_matrix - wtLenDist_matrix.min())/(wtLenDist_matrix.max() - wtLenDist_matrix.min())
    stLenDist_matrix = (stLenDist_matrix - stLenDist_matrix.min())/(stLenDist_matrix.max() - stLenDist_matrix.min())
    
    attribution = (tfidf_matrix, sparce_matrix, wtLenDist_matrix, stLenDist_matrix, normalizedLabels) 
    attribution = np.concatenate(attribution, axis = 1)
    
    pred_svm = svm.predict(attribution)
    
    return pred_svm

###

import PySimpleGUI as sg

sg.theme('BluePurple')

layout = [  [sg.Text('Köşe yazısını giriniz.')],
            [sg.Multiline(size=(50, 20))],
            [sg.Button('Yazarı bul')],
            [sg.Text('Yazar: '), sg.Text(size=(15,1), key='-OUTPUT-')]]


window = sg.Window('Yazar Bulma', layout)

while True:
    event, values = window.read()
    
    result = processing(values[0])
    
    if event == 'Yazarı bul':
        # Update the "output" text element to be the value of "input" element
        window['-OUTPUT-'].update(mapping[result[0]])
    
    if event == sg.WIN_CLOSED:
        break
    print('Author: ', mapping[result], result)

window.close()

###


