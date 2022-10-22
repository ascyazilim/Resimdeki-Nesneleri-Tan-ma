import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

df = pd.read_csv('NLPlabeledData.tsv', delimiter="\t", quoting=3)

ilkbes = df.head()
len(df)
len(df["review"])

nltk.download('stopwords')

yorum = df.review[0]
#Html temizleme
yorum = BeautifulSoup(yorum).get_text()
#noktalama işareteri -regex   küçük a-z ye büyük A-Z dışındaki herşey yerine boşluk
yorum = re.sub("[^a-zA-Z]",' ',yorum)
#kucuk harfe cevirme
yorum = yorum.lower()
#stopwords listeye cevir
yorum = yorum.split()
swords = set(stopwords.words("english"))
yorum = [w for w in yorum if w not in swords]  #w = word sword içindeyse alma değilse al
len(yorum)   #437'den 219'a düştü
 

#tüm veriseti temizleme işlemi

def process(review):
    #html temizleme
    review = BeautifulSoup(review).get_text()
    #noktalama
    review = re.sub("[^a-zA-Z]",' ',review)
    #kucuk harf
    review = review.lower()
    #stopwords
    review = review.split()
    swords = set(stopwords.words("english"))
    review = [w for w in review if w not in swords]
    # space ile birleştirme
    return(" ".join(review))

train_x_tum = []
for i in range(len(df["review"])):
    train_x_tum.append(process(df["review"][i]))


x = train_x_tum
y = np.array(df["sentiment"])

train_x, test_x, y_train, y_test = train_test_split(x,y, test_size= 0.1, random_state = 42)


vectorizer = CountVectorizer(max_features= 5000)

train_x = vectorizer.fit_transform(train_x)

train_x = train_x.toarray()

train_y = y_train

train_x.shape, train_y.shape   #(22500, 5000), (22500,)


model = RandomForestClassifier(n_estimators= 100) #ağaç sayısı 
model.fit(train_x, train_y)


test_xx = vectorizer.transform(test_x)
test_xx = test_xx.toarray()
test_xx.shape    #(2500,5000)
 

test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(y_test, test_predict)

print("Doğruluk oranı : % ", dogruluk * 100)


















