import pyaudio
import wave
import soundfile
import librosa
import sounddevice as sd
import numpy as np
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from y_audio_utils import read_sounfile, extract_feature, aug_speed, aug_add_noise, aug_shift_zero, extract_feature2

le = LabelEncoder()

def load_data(test_size = 0.2):
    x, y = [], []
    empty_files = []
    for base_path in glob.glob("Dataset_04_07_2020\Dataset\speaker\G*"):
        print(base_path.split("\\")[2])
        for file in glob.glob(base_path + "\*.wav"):
            basename = os.path.basename(file)   # get the base name of the audio file
            print("Grupo " + base_path)
            keyword = basename.split("_")[1]
            print(keyword)
            #print(base_path.split("\\")[3] + "-" + keyword)
            # remove empty files (G1)
            sound_file = soundfile.SoundFile(file)
            if len(sound_file.read(dtype='float32')) == 0:
                print("Empty File : " + file)
                empty_files.append(file)
                continue
            # Raw wave
            sound_frame, sr = read_sounfile(file)
            #sd.play(sound_frame, sr)
            features = extract_feature2(sound_frame,sr,mfcc=True,centroid=True)
            print(len(features))
            x.append(features)
            y.append(keyword)
            # Time Shift with padding
            frame_shift = aug_shift_zero(sound_frame,sr,0.2,shift_direction='both')
            #sd.play(frame_shift, sr)
            #sd.play(frame_shift, sr)
            features = extract_feature2(frame_shift, sr, mfcc=True,centroid=True)
            print(len(features))
            x.append(features)
            y.append(keyword)
            # Add Noise
                #frame_noise = aug_add_noise(sound_frame)
                #features = extract_feature(frame_noise, sr, mfcc=True, chroma=True, mel=True)
                #x.append(features)
                #y.append(keyword)
            # Pitch
                # frame_pitch = aug_pitch(sound_frame,sr,1.2)
                # features = extract_feature(frame_pitch, sr, mfcc=True, chroma=True, mel=True)
                # x.append(features)
                # y.append(keyword)
            # Speed Slower
            frame_slower = aug_speed(sound_frame,0.9)
            #sd.play(frame_slower, sr)
            features = extract_feature2(frame_slower, sr, mfcc=True,centroid=True)
            print(len(features))
            x.append(features)
            y.append(keyword)
            # Speed Faster
            frame_faster = aug_speed(sound_frame,1.1)
            #sd.play(frame_faster, sr)
            features = extract_feature2(frame_faster, sr, mfcc=True,centroid=True)
            print(len(features))
            x.append(features)
            y.append(keyword)
    le.fit(y)
    list(le.classes_)
    yt=le.transform(y)
    return train_test_split(x, y, test_size=test_size, stratify=y,random_state=True)

X_train, X_test, Y_train, Y_test = load_data(test_size=0.25)

#https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

transformer = RobustScaler().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

if not os.path.isdir("../classifier_out/utils_audio"):
    os.mkdir("../classifier_out/utils_audio")
pickle.dump(transformer, open('utils_audio/scaler_keyword_robust_aug.bin','wb'))

print("[+] Number of training samples:", X_train.shape[0]) # number of samples in training data
print("[+] Number of testing samples:", X_test.shape[0]) # number of samples in testing data
print("[+] Number of features:", X_train.shape[1]) # number of features used, this is a vector of features extracted using extract_features() function

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300, 150,50), learning_rate='adaptive',max_iter=500)

print("[*] Training the model...")
model.fit(X_train,Y_train)
#clf = OneVsRestClassifier(model)
#clf= clf.fit(X_train, Y_train)

# predict 25% of data to measure how good we are
Y_predict = model.predict(X_test)
#Y_predict = clf.predict(X_test)


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
# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("../classifier_out/utils_audio"):
    os.mkdir("../classifier_out/utils_audio")
pickle.dump(model, open("../classifier_out/utils_audio/classifier_keyword_aug.model", "wb"))
print(le.classes_)
stop=0