import cv2
import numpy as np
import video
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pickle

#image = cv2.imread("Dataset/MEEC_1920_LI2-master/G8/Imagens/G8_1.jpg")
#cv2.imshow("Imagem G1_0",image)

vid = video.Video()

XX=np.zeros((8*20,68,2), dtype=float)
YY=np.zeros((8*20), dtype = int)


for i in range(1,9):
    ims=[]
    for j in range(0,20):
        print(i,j);
        ims.append(cv2.imread('Dataset/images/G%d_%d.jpg'%(i,j)))
        im_landmarks, shape = vid.getLandmaks(ims[j])
        XX[((i-1)*20+j),:]=shape
        YY[((i-1)*20+j)]=i

print(XX)
print(YY)

nsamples,nx,ny = XX.shape
XX=XX.reshape((nsamples,nx*ny))
##nsamples,nx,ny = YY.shape
#YY=YY.reshape((nsamples,nx*ny))

X_train, X_test, Y_train, Y_test = train_test_split(XX,YY, test_size=0.2, stratify = YY, random_state=True)
print(type(X_train))
print(X_train)
print(Y_train)
model = KNeighborsClassifier(n_neighbors=3)
#model = MLPClassifier(hidden_layer_sizes=(10, 10), learning_rate_init=0.0001, max_iter=10000)

model.fit(X_train,Y_train)
pickle.dump(model, open('KNeighborsClassifier.sav', 'wb'))

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
