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

# XX=np.zeros((8*20,68*2), dtype=float) #68,2)
# XX=np.zeros((8*20,67), dtype=float) #68,2)
nGrupos = 8;
n_imagesForGroup=20;
nLandmarks=68;


'''
XX1=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX2=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX3=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX4=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX5=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX6=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX7=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX8=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)





YY1=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY2=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY3=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY4=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY5=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY6=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY7=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY8=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)

Counter1=0;
Counter2=0;
Counter3=0;
Counter4=0;
Counter5=0;
Counter6=0;
Counter7=0;
Counter8=0;


for i in range(1,nGrupos+1):
    for j in range(0,n_imagesForGroup):
        ims = []
        for h in range(0, nGrupos-1):

            print(i, j, h);
            ims.append(cv2.imread('Dataset/images/G%d_%d_%d.jpg' % (i, j, h)))

            shape = vid.training(ims[h])

            #XX[((i - 1) * n_imagesForGroup + j*nGrupos), :] = shape.reshape(-1)  # np.concatenate((shape.reshape(-1),shape_face),axis=None)

            if i == 1:
                if(h==0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 1
                    YY2[Counter2] = 0
                    YY3[Counter3] = 0
                    YY4[Counter4] = 0
                    YY5[Counter5] = 0
                    YY6[Counter6] = 0
                    YY7[Counter7] = 0
                    YY8[Counter8] = 0
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX1[Counter1, :] = shape.reshape(-1)
                    YY1[Counter1] = 1
                    Counter1 = Counter1 + 1


            elif i == 2:
                if (h == 0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 0
                    YY2[Counter2] = 1
                    YY3[Counter3] = 0
                    YY4[Counter4] = 0
                    YY5[Counter5] = 0
                    YY6[Counter6] = 0
                    YY7[Counter7] = 0
                    YY8[Counter8] = 0
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX2[Counter2, :] = shape.reshape(-1)
                    YY2[Counter2] = 1
                    Counter2 = Counter2 + 1
            elif i == 3:
                if (h == 0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 0
                    YY2[Counter2] = 0
                    YY3[Counter3] = 1
                    YY4[Counter4] = 0
                    YY5[Counter5] = 0
                    YY6[Counter6] = 0
                    YY7[Counter7] = 0
                    YY8[Counter8] = 0
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX3[Counter3, :] = shape.reshape(-1)
                    YY3[Counter3] = 1
                    Counter3 = Counter3 + 1

            elif i == 4:
                if (h == 0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 0
                    YY2[Counter2] = 0
                    YY3[Counter3] = 0
                    YY4[Counter4] = 1
                    YY5[Counter5] = 0
                    YY6[Counter6] = 0
                    YY7[Counter7] = 0
                    YY8[Counter8] = 0
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX4[Counter4, :] = shape.reshape(-1)
                    YY4[Counter4] = 1
                    Counter4 = Counter4 + 1
            elif i == 5:
                if (h == 0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 0
                    YY2[Counter2] = 0
                    YY3[Counter3] = 0
                    YY4[Counter4] = 0
                    YY5[Counter5] = 1
                    YY6[Counter6] = 0
                    YY7[Counter7] = 0
                    YY8[Counter8] = 0
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX5[Counter5, :] = shape.reshape(-1)
                    YY5[Counter5] = 1
                    Counter5 = Counter5 + 1
            elif i == 6:
                if (h == 0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 0
                    YY2[Counter2] = 0
                    YY3[Counter3] = 0
                    YY4[Counter4] = 0
                    YY5[Counter5] = 0
                    YY6[Counter6] = 1
                    YY7[Counter7] = 0
                    YY8[Counter8] = 0
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX6[Counter6, :] = shape.reshape(-1)
                    YY6[Counter6] = 1
                    Counter6 = Counter6 + 1
            elif i == 7:
                if (h == 0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 0
                    YY2[Counter2] = 0
                    YY3[Counter3] = 0
                    YY4[Counter4] = 0
                    YY5[Counter5] = 0
                    YY6[Counter6] = 0
                    YY7[Counter7] = 1
                    YY8[Counter8] = 0
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX7[Counter7, :] = shape.reshape(-1)
                    YY7[Counter7] = 1
                    Counter7 = Counter7 + 1
            elif i == 8:
                if (h == 0):
                    XX1[Counter1, :] = shape.reshape(-1)
                    XX2[Counter2, :] = shape.reshape(-1)
                    XX3[Counter3, :] = shape.reshape(-1)
                    XX4[Counter4, :] = shape.reshape(-1)
                    XX5[Counter5, :] = shape.reshape(-1)
                    XX6[Counter6, :] = shape.reshape(-1)
                    XX7[Counter7, :] = shape.reshape(-1)
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY1[Counter1] = 0
                    YY2[Counter2] = 0
                    YY3[Counter3] = 0
                    YY4[Counter4] = 0
                    YY5[Counter5] = 0
                    YY6[Counter6] = 0
                    YY7[Counter7] = 0
                    YY8[Counter8] = 1
                    Counter1 = Counter1 + 1
                    Counter2 = Counter2 + 1
                    Counter3 = Counter3 + 1
                    Counter4 = Counter4 + 1
                    Counter5 = Counter5 + 1
                    Counter6 = Counter6 + 1
                    Counter7 = Counter7 + 1
                    Counter8 = Counter8 + 1
                else:
                    XX8[Counter8, :] = shape.reshape(-1)
                    YY8[Counter8] = 1
                    Counter8 = Counter8 + 1


#nsamples,nx,ny = XX.shape
#XX=XX.reshape((nsamples,nx*ny))

##nsamples,nx,ny = YY.shape
#YY=YY.reshape((nsamples,nx*ny))

pickle.dump(XX1, open('shapeX1.sav', 'wb'))
pickle.dump(XX2, open('shapeX2.sav', 'wb'))
pickle.dump(XX3, open('shapeX3.sav', 'wb'))
pickle.dump(XX4, open('shapeX4.sav', 'wb'))
pickle.dump(XX5, open('shapeX5.sav', 'wb'))
pickle.dump(XX6, open('shapeX6.sav', 'wb'))
pickle.dump(XX7, open('shapeX7.sav', 'wb'))
pickle.dump(XX8, open('shapeX8.sav', 'wb'))
pickle.dump(YY1, open('shapeY1.sav', 'wb'))
pickle.dump(YY2, open('shapeY2.sav', 'wb'))
pickle.dump(YY3, open('shapeY3.sav', 'wb'))
pickle.dump(YY4, open('shapeY4.sav', 'wb'))
pickle.dump(YY5, open('shapeY5.sav', 'wb'))
pickle.dump(YY6, open('shapeY6.sav', 'wb'))
pickle.dump(YY7, open('shapeY7.sav', 'wb'))
pickle.dump(YY8, open('shapeY8.sav', 'wb'))



'''
XX1=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX2=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX3=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX4=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX5=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX6=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX7=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
XX8=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup,nLandmarks*nLandmarks-nLandmarks), dtype=float)
YY1=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY2=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY3=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY4=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY5=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY6=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY7=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)
YY8=np.zeros((nGrupos*n_imagesForGroup+(nGrupos-2)*n_imagesForGroup), dtype = int)

XX1[:, :] = pickle.load(open('shapeX1.sav', 'rb'))
XX2[:, :] = pickle.load(open('shapeX2.sav', 'rb'))
XX3[:, :] = pickle.load(open('shapeX3.sav', 'rb'))
XX4[:, :] = pickle.load(open('shapeX4.sav', 'rb'))
XX5[:, :] = pickle.load(open('shapeX5.sav', 'rb'))
XX6[:, :] = pickle.load(open('shapeX6.sav', 'rb'))
XX7[:, :] = pickle.load(open('shapeX7.sav', 'rb'))
XX8[:, :] = pickle.load(open('shapeX8.sav', 'rb'))
YY1[:] = pickle.load(open('shapeY1.sav', 'rb'))
YY2[:] = pickle.load(open('shapeY2.sav', 'rb'))
YY3[:] = pickle.load(open('shapeY3.sav', 'rb'))
YY4[:] = pickle.load(open('shapeY4.sav', 'rb'))
YY5[:] = pickle.load(open('shapeY5.sav', 'rb'))
YY6[:] = pickle.load(open('shapeY6.sav', 'rb'))
YY7[:] = pickle.load(open('shapeY7.sav', 'rb'))
YY8[:] = pickle.load(open('shapeY8.sav', 'rb'))



X_train1, X_test1, Y_train1, Y_test1 = train_test_split(XX1, YY1, test_size=0.2, stratify=YY1, random_state=True)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(XX2, YY2, test_size=0.2, stratify=YY2, random_state=True)
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(XX3, YY3, test_size=0.2, stratify=YY3, random_state=True)
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(XX4, YY4, test_size=0.2, stratify=YY4, random_state=True)
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(XX5, YY5, test_size=0.2, stratify=YY5, random_state=True)
X_train6, X_test6, Y_train6, Y_test6 = train_test_split(XX6, YY6, test_size=0.2, stratify=YY6, random_state=True)
X_train7, X_test7, Y_train7, Y_test7 = train_test_split(XX7, YY7, test_size=0.2, stratify=YY7, random_state=True)
X_train8, X_test8, Y_train8, Y_test8 = train_test_split(XX8, YY8, test_size=0.2, stratify=YY8, random_state=True)

scaler = StandardScaler()
scaler.fit(X_train1)
X_train1 = scaler.transform(X_train1)
X_test1 = scaler.transform(X_test1)
X_train2 = scaler.transform(X_train2)
X_test2 = scaler.transform(X_test2)
X_train3 = scaler.transform(X_train3)
X_test3 = scaler.transform(X_test3)
X_train4 = scaler.transform(X_train4)
X_test4 = scaler.transform(X_test4)
X_train5 = scaler.transform(X_train5)
X_test5 = scaler.transform(X_test5)
X_train6 = scaler.transform(X_train6)
X_test6 = scaler.transform(X_test6)
X_train7 = scaler.transform(X_train7)
X_test7 = scaler.transform(X_test7)
X_train8 = scaler.transform(X_train8)
X_test8 = scaler.transform(X_test8)

pickle.dump(scaler, open('Scaler.sav', 'wb'))


# model = SVC(C=25.0,gamma=0.00001)  #C=25.0,gamma=0.0001)
# model = KNeighborsClassifier(n_neighbors=3)
# model = DecisionTreeClassifier(max_depth=50)
# model = RandomForestClassifier(n_estimators=100)
# model = AdaBoostClassifier(n_estimators=1000)
# model = MLPClassifier(hidden_layer_sizes=(300, 150,50), learning_rate_init=0.001, max_iter=300,shuffle=False)

print("################################### 1 #############################")
model = LinearSVC(max_iter=10000, dual=False, fit_intercept=False)

model.fit(X_train1, Y_train1)

Y_predict = model.predict(X_test1)

cm = metrics.confusion_matrix(Y_test1, Y_predict, labels=[0, 1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test1, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test1, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test1, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC1.sav', 'wb'))


print("################################### 2 #############################")
model = LinearSVC(max_iter=10000, dual=False, fit_intercept=True)

model.fit(X_train2, Y_train2)

Y_predict = model.predict(X_test2)

cm = metrics.confusion_matrix(Y_test2, Y_predict, labels=[0,1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test2, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test2, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test2, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC2.sav', 'wb'))

print("################################### 3 #############################")

model = LinearSVC(max_iter=10000, dual=False, fit_intercept=False)

model.fit(X_train3, Y_train3)

Y_predict = model.predict(X_test3)

cm = metrics.confusion_matrix(Y_test3, Y_predict, labels=[0,1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test3, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test3, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test3, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC3.sav', 'wb'))

print("################################### 4 #############################")

model = LinearSVC(max_iter=10000, dual=False, fit_intercept=False)

model.fit(X_train4, Y_train4)

Y_predict = model.predict(X_test4)

cm = metrics.confusion_matrix(Y_test4, Y_predict, labels=[0,1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test4, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test4, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test4, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC4.sav', 'wb'))

print("################################### 5 #############################")

model = LinearSVC(max_iter=10000, dual=False, fit_intercept=False)

model.fit(X_train5, Y_train5)

Y_predict = model.predict(X_test5)

cm = metrics.confusion_matrix(Y_test5, Y_predict, labels=[0,1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test5, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test5, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test5, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC5.sav', 'wb'))

print("################################### 6 #############################")

model = LinearSVC(max_iter=10000, dual=False, fit_intercept=False)

model.fit(X_train6, Y_train6)

Y_predict = model.predict(X_test6)

cm = metrics.confusion_matrix(Y_test6, Y_predict, labels=[0,1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test6, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test6, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test6, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC6.sav', 'wb'))

print("################################### 7 #############################")

model = LinearSVC(max_iter=10000, dual=False, fit_intercept=False)

model.fit(X_train7, Y_train7)

Y_predict = model.predict(X_test7)

cm = metrics.confusion_matrix(Y_test7, Y_predict, labels=[0,1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test7, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test7, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test7, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC7.sav', 'wb'))

print("################################### 8 #############################")

model = LinearSVC(max_iter=10000, dual=False, fit_intercept=False)

model.fit(X_train8, Y_train8)

Y_predict = model.predict(X_test8)

cm = metrics.confusion_matrix(Y_test8, Y_predict, labels=[0,1])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test8, Y_predict)
print("Precision Recall Fscor Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test8, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test8, Y_predict)
print("Classification Report:")
print(cr)

pickle.dump(model, open('LinearSVC8.sav', 'wb'))