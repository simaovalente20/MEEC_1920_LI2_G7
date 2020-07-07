import cv2
import numpy as np
import video
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import pickle



vid = video.Video()


#XX=np.zeros((8*20,68*2), dtype=float) #68,2)
#XX=np.zeros((8*20,67), dtype=float) #68,2)
nGrupos=8;

'''


XX=np.zeros((nGrupos*20,68*68-68), dtype=float)
YY=np.zeros((nGrupos*20), dtype = int)

for i in range(1,nGrupos+1):
    ims=[]
    for j in range(0,20):
        print(i,j);
        ims.append(cv2.imread('Dataset/images/G%d_%d.jpg'%(i,j)))

        shape = vid.training(ims[j])

        XX[((i-1)*20+j),:]=shape.reshape(-1) #np.concatenate((shape.reshape(-1),shape_face),axis=None)

        YY[((i - 1) * 20 + j)] = i
        #if i==7 :
        #    YY[((i-1)*20+j)]=1
        #else:
        #    YY[((i - 1) * 20 + j)] = 0
        

print(XX)
print(YY)


#nsamples,nx,ny = XX.shape
#XX=XX.reshape((nsamples,nx*ny))

##nsamples,nx,ny = YY.shape
#YY=YY.reshape((nsamples,nx*ny))

pickle.dump(XX, open('shapeX.sav', 'wb'))
pickle.dump(YY, open('shapeY.sav', 'wb'))

'''
        
XX=np.zeros((nGrupos*20,68*68-68), dtype=float) #68,2)
YY=np.zeros((nGrupos*20), dtype = int)
XX[:,:] = pickle.load(open('../shapeX.sav', 'rb'))
YY[:] = pickle.load(open('../shapeY.sav', 'rb'))


X_train, X_test, Y_train, Y_test = train_test_split(XX,YY, test_size=0.2, stratify = YY, random_state=True)

scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

model = LinearSVC(max_iter=5000,dual=False,fit_intercept=False)
#model = SVC(C=25.0,gamma=0.00001)  #C=25.0,gamma=0.0001)
#model = KNeighborsClassifier(n_neighbors=3)
#model = DecisionTreeClassifier(max_depth=50)
#model = RandomForestClassifier(n_estimators=100)
#model = AdaBoostClassifier(n_estimators=1000)
#model = MLPClassifier(hidden_layer_sizes=(300, 150,50), learning_rate_init=0.001, max_iter=300,shuffle=False)

model.fit(X_train,Y_train)
pickle.dump(model, open('../LinearSVC.sav', 'wb'))
pickle.dump(scaler, open('../Scaler.sav', 'wb'))

Y_predict = model.predict(X_test)

cm= metrics.confusion_matrix(Y_test, Y_predict, labels=[1,2,3,4,5,6,7,8])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test,Y_predict)
print("Accuracy:")
print(accuracy)

cr=metrics.classification_report(Y_test,Y_predict)
print("Classification Report:")
print(cr)

print(model.decision_function(X_test))
print(model._predict_proba_lr(X_test))
