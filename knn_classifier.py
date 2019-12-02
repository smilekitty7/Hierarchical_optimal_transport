# build classifer
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import operator
import numpy as np
from collections import Counter


def knnclassify(test,x_train,y_train,k,method):
    distances=[]
    for x in range(len(x_train)):
        distances.append(method(test, x_train[x]))
    #print("dist",distances)
    #print("hhhhhhhhhhhhhhhhhh")
    neighbor_inx=np.argsort(distances)[:k]
    neighbors = []
    for x in range(len(neighbor_inx)):
        
        neighbors.append(y_train[neighbor_inx[x]])  
    
#    classvote={}
#    nclass = np.unique(neighbors)
#    for i in range(len(nclass)):
#        temp=neighbors.count(nclass[i])
#        classvote.append(temp)
#        
#    predict=np.argmax(classvote)
#    
    most_common = Counter(neighbors).most_common(1)
    return most_common[0][0]

def myknn(k,x_train,y_train,x_test,y_test,method):
    y_pred=[]
    for i in range(len(x_test)):
        ypred=knnclassify(x_test[i],x_train,y_train,k,method)
        y_pred.append(ypred)
    
    count=0
    for i in range(len(y_test)):
        if y_pred[i]==y_test[i]:
            count=count+1
            
            
    test_error=1-count/len(y_test)
    
    return test_error
        






#create own knn with our designed distance
#k is neighbors number

        





        
