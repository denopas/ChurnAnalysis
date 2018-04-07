import numpy as np
#sonuçların yeniden üretilebilir olmasını amaçlıyoruz
np.random.seed(1)   
from sklearn import cross_validation
from sklearn import preprocessing
import pandas as pd

columns = [
    'state',
    'account length', 
    'area code', 
    'phone number', 
    'international plan', 
    'voice mail plan', 
    'number vmail messages',
    'total day minutes',
    'total day calls',
    'total day charge',
    'total eve minutes',
    'total eve calls',
    'total eve charge',
    'total night minutes',
    'total night calls',
    'total night charge',
    'total intl minutes',
    'total intl calls',
    'total intl charge',
    'number customer service calls',
    'churn']

data = pd.read_csv('churn.data.txt', header = None, names = columns)
#Datasetin orjinali hali
print("Dataset orjinal hali: " + str(data.shape))

#Preprocessing Adım 1: yes, no, true, false mapping
mapping = {'no': 0., 'yes':1., 'False.':0., 'True.':1.}
data.replace({'international plan' : mapping, 'voice mail plan' : mapping, 'churn' : mapping}, regex = True, inplace = True)

#Preprocessing Adım 2: phone number, area code, state özniteliklerinin kaldırılması
data.drop('phone number', axis = 1, inplace = True)
data.drop('area code', axis = 1, inplace = True)
data.drop('state', axis = 1, inplace = True)
print("Dataset preprocessing sonrasi: " + str(data.shape))

print("Veri turleri:")
print(data.dtypes)

#Yorum 1: Histogram grafiği görmek için aşağıdaki kod satırlarını kullanabilirsiniz
'''
import matplotlib.pyplot as plt
num_bins = 10
data.hist(bins=num_bins, figsize=(20,15))

plt.savefig("churn_histogram_plots")
plt.show()
'''
data1 = data[data['churn']==1]
print("Churn olanlar-data1:"+ str(data1.shape))

data2 = data[data['churn']==0]
print("Churn olmayanlar-data2:"+ str(data2.shape))

#Her iki sinifta da esit sayida ornek olmasini istiyoruz
data = data1.append(data2[:483])
print("Son veriseti :"+ str(data.shape))


#Egitim  ve test verisini parcaliyoruz --> 80% / 20%
X = data.ix[:, data.columns != 'churn']
Y = data['churn']

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Ölçeklendirme
scaler = preprocessing.MinMaxScaler((-1,1))
scaler.fit(X)

XX_train = scaler.transform(X_train.values)
XX_test  = scaler.transform(X_test.values)

YY_train = Y_train.values 
YY_test  = Y_test.values

print (X_train.shape, YY_train.shape)
print (X_test.shape, YY_test.shape)

# Sınıflandırma Modellerine Ait Kütüphaneler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

# Modelleri Hazırlayalım
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('AdaBoostClassifier', AdaBoostClassifier(learning_rate=0.5)))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier())) 

from sklearn.metrics import classification_report
# Modelleri test edelim
for name, model in models:
    model = model.fit(XX_train, YY_train)
    Y_pred = model.predict(XX_test)
    from sklearn import metrics
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(YY_test, Y_pred)*100))
    #Yorum 3: Confusion matris görmek için aşağıdaki kod satırlarını kullanabilirsiniz
    '''report = classification_report(YY_test, Y_pred)
    print(report)'''
