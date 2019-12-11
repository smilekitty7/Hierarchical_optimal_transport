# build classifer
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

def knnclassify(test,x_train,y_train,D,k,method):
    distances=[]
    for x in range(len(x_train))):
        dist = method(test, x_train[x],D)
        distances.append(dist)
  

    neighbor_inx=np.argsort(distances)[:k]
    neighbors = []
    for x in range(len(neighbor_inx)):
        
        neighbors.append(y_train[neighbor_inx[x]])  
  
    most_common = Counter(neighbors).most_common(1)
    return most_common[0][0]

def myknn(k,x_train,y_train,x_test,y_test,D,method):
    y_pred=[]
    tempcount = 0
    for i in range(len(x_test)):
        ypred_t=knnclassify(x_test[i],x_train,y_train,D,k,method)
        y_pred.append(ypred_t)
        tempcount = tempcount+1
        print("count: ", tempcount)
    count=0
    for i in range(len(y_test)):
        if y_pred[i]==y_test[i]:
            count=count+1
            
            
    test_error=1-count/len(y_test)
    
    return test_error
        






#create own knn with our designed distance
#k is neighbors number

        





        
