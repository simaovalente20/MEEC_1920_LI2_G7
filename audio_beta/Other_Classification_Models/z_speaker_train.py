import pyaudio
import wave
import soundfile
import librosa
import numpy as np
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump,load
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from y_audio_utils import read_sounfile, extract_feature, aug_speed, aug_add_noise

model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}

def load_data(test_size = 0.2):
    x, y = [], []
    empty_files = []
    for base_path in glob.glob("Dataset\speaker\G*"):
        #print(base_path.split("\\")[1])
        for file in glob.glob(base_path + "\*.wav"):
            basename = os.path.basename(file)   # get the base name of the audio file
            #print("Grupo " + base_path)
            speaker = base_path.split("\\")[2]
            print(speaker)
            # remove empty files (G1)
            sound_file = soundfile.SoundFile(file)
            if len(sound_file.read(dtype='float32')) == 0:
                print("Empty File : " + file)
                empty_files.append(file)
                continue
            sound_frame, sr = read_sounfile(file)
            features = extract_feature(sound_frame, sr, mfcc=True, chroma=True, mel=True)
            #features = extract_feature(file,mfcc=True, chroma=True, mel=True)
            x.append(features)
            y.append(speaker)
    return train_test_split(np.array(x), y, test_size=test_size,random_state=9)


X_train, X_test, Y_train, Y_test = load_data(test_size=0.25)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if not os.path.isdir("../classifier_out/utils_audio"):
    os.mkdir("../classifier_out/utils_audio")
pickle.dump(scaler,open('utils_audio/scaler_speaker.bin','wb'))


print("[+] Number of training samples:", X_train.shape[0])
print("[+] Number of testing samples:", X_test.shape[0])
print("[+] Number of features:", X_train.shape[1])

model = MLPClassifier(**model_params)

print("[*] Training the model...")
clf = OneVsRestClassifier(model)

clf = clf.fit(X_train,Y_train)
#model.fit(X_train,Y_train)

# predict 25% of data to measure how good we are
#Y_predict = model.predict(X_test)
Y_predict = clf.predict(X_test)

cm= metrics.confusion_matrix(Y_test, Y_predict)
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

# save the model
if not os.path.isdir("../classifier_out/utils_audio"):
    os.mkdir("../classifier_out/utils_audio")
pickle.dump(clf, open("utils_audio/classifier_speaker_OvR.model", "wb"))

stop=0