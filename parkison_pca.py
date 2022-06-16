import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
print('yes')

















##################################################
data = pd.read_csv("../input/speech-park/pd_speech_features.csv")
data.head()
#print(data.shape)












X0=data.drop('gender',axis=1)
X=X0.drop('class',axis=1)
Y=data['class']
X=np.array(X)
Y=np.array(Y)
#print(Y)
print(X.shape)
print(Y.shape)
X_data=X
Y_data=Y





















X_data = preprocessing.scale(X_data)
print(X_data.shape)
print(Y_data.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.5)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)















def data_classifiers(X_train, Y_train, X_test, Y_test):
    model1=LogisticRegression()
    model1.fit(X_train,Y_train)
    test_score1 = model1.score(X_test,Y_test)
    
    model2=SVC()
    model2.fit(X_train,Y_train)
    test_score2 = model2.score(X_test,Y_test)
    
    model3=KNeighborsClassifier()
    model3.fit(X_train,Y_train)
    test_score3 = model3.score(X_test,Y_test)
    
    model4=RandomForestClassifier()
    model4.fit(X_train,Y_train)
    test_score4 = model4.score(X_test,Y_test)
    
    NAMES  = np.array(['LR', 'SVM', 'KNN', 'RF'])
    FLOATS = np.array([ test_score1*100,test_score2*100,test_score3*100,test_score4*100])
    DAT =  np.column_stack((NAMES, FLOATS))
    return DAT, FLOATS 












DAT, FLOATS=data_classifiers(X_train, Y_train, X_test, Y_test)
print(DAT)










np.save('parkinson_classifier_result_vocal.npy',DAT)




np.savetxt('parkinson_classifier_result_vocal.npy', FLOATS, delimiter =', ')












from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_data = pca.fit_transform(X_data)
print(X_data.shape)
#principalDf = pd.DataFrame(data = pc, columns = ['principal 1', 'principal 2'])
#principalDf.head()
#print(principalDf)

X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.5)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

DAT, FLOATS=data_classifiers(X_train, Y_train, X_test, Y_test)
print(DAT)