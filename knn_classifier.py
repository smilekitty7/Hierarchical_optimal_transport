# build classifer
import numpy as np

def knnclassify(test,x_train,y_train,k,method):
    distances = []
    
    for x in range(len(x_train)):
        dist = method(x_train[x], y_train[x])
        distances.append(dist)
    sortindices=np.argsort(dist)
    neighbors = []
    for x in range(k):
        neighbors.append(x_train[sortindices[x]])  #find k nearest neighbors
    
    classvote={}#count the frequency
    nclass = np.unique(neighbors)
    for i in range(len(nclass)):
        temp=neighbors.count(nclass[1])
        classvote.append(temp)
        
    predict=np.argmax(classvote)
    
    return nclass[predict]#return the most popular one 

def myknn(k,x_train,y_train,x_test,y_test,method):
    y_pred=[]
    for i in range(len(x_test)):
        ypred=knnclassify(x_test[i],x_train,y_train,k,method)
        y_pred.append(ypred)
    
    count=0
    for i in range(len(y_test)):
        if y_pred==y_test:
            count=count+1
            
            
    test_error=1-count/len(y_test)
    
    return test_error
        
