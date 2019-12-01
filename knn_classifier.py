# build classifer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#create own knn with our designed distance
#k is neighbors number
def myknn(k,x_train,y_train,x_test,y_test,method):
    knn=KNeighborsClassifier(n_neighbors=k,weights=method)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    
