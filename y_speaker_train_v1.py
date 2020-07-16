import pyaudio
import wave
import soundfile
import librosa
import numpy as np
import os, glob, pickle
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from y_audio_utils import read_sounfile, extract_feature, aug_speed, aug_add_noise, aug_shift_zero, extract_feature2,aug_shift,extract_feature3
import matplotlib.pyplot as plt

def load_data(test_size = 0.2):
    x, y = [], []
    empty_files = []
    for base_path in glob.glob("Dataset_04_07_2020\Dataset\speaker\G*"):#Dataset\speaker\G*"):
        print("###################" + base_path.split("\\")[2])
        for file in glob.glob(base_path + "\*.wav"):
            basename = os.path.basename(file)   # get the base name of the audio file
            #print("Grupo " + base_path)
            speaker = base_path.split("\\")[3]
            print(speaker)
            # remove empty files (G1)
            sound_file = soundfile.SoundFile(file)
            if len(sound_file.read(dtype='float32')) == 0:
                print("Empty File : " + file)
                empty_files.append(file)
                continue
            # Raw wave
            sound_frame, sr = read_sounfile(file)
            #plt.plot(sound_frame)
            #plt.show()
            sound_clipped = librosa.util.fix_length(sound_frame, sr * 2)
            # Add Noise
            sound_2s = aug_add_noise(sound_clipped)
            trimmed, index = librosa.effects.trim(sound_2s, top_db=30)
            print(librosa.get_duration(sound_2s,sr), librosa.get_duration(trimmed,sr))
            # sd.play(sound_frame, sr)
            features = extract_feature3(sound_2s, sr, mfcc=True,chroma=True)
            print(len(features))
            x.append(features)
            y.append(speaker)
            i = 1
            '''for i in range(1, 8):
                frame_shift = aug_shift(trimmed, sr, i)
                print(librosa.get_duration(sound_2s,sr), librosa.get_duration(trimmed,sr))
                features = extract_feature3(frame_shift, sr, mfcc=True,chroma=True)
                print(len(features))
                x.append(features)
                y.append(speaker)'''
            # Add Noise
                #frame_noise = aug_add_noise(sound_frame)
                #features = extract_feature(frame_noise, sr, mfcc=True, chroma=True, mel=True)
                #x.append(features)
                #y.append(speaker)
            # Pitch
                # frame_pitch = aug_pitch(sound_frame,sr,1.2)
                # features = extract_feature(frame_pitch, sr, mfcc=True, chroma=True, mel=True)
                # x.append(features)
                # y.append(speaker)
            # Speed Slower
                # frame_slower = aug_speed(sound_frame,0.9)
                # features = extract_feature_speaker(frame_slower, sr, mfcc=True,chroma=True)
                # x.append(features)
                # y.append(speaker)
            # Speed Faster
                # frame_faster = aug_speed(sound_frame,1.1)
                # features = extract_feature_speaker(frame_faster, sr, mfcc=True,chroma=True)
                # x.append(features)
                # y.append(speaker)
    return train_test_split(np.array(x), y, test_size=0.2, stratify = y, random_state=True)

X_train, X_test, Y_train, Y_test = load_data(test_size=0.25)

#https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

transformer = RobustScaler().fit(X_train)
#X_train = transformer.transform(X_train)
#X_test = transformer.transform(X_test)

if not os.path.isdir("utils_mfcc"):
    os.mkdir("utils_mfcc")
pickle.dump(scaler, open('utils_mfcc/scaler_speaker_aug_13mfcc.bin', 'wb'))
pickle.dump(transformer, open('utils_mfcc/scaler_speaker_robust_aug_13mfcc.bin', 'wb'))

print("[+] Number of training samples:", X_train.shape[0]) # number of samples in training data
print("[+] Number of testing samples:", X_test.shape[0]) # number of samples in testing data
print("[+] Number of features:", X_train.shape[1]) # number of features used, this is a vector of features extracted using extract_features() function
model = MLPClassifier(alpha=0.0001, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300), learning_rate='adaptive', max_iter=500)

print("[*] Training the model...")
model.fit(X_train,Y_train)
#clf = OneVsRestClassifier(model)
#clf= clf.fit(X_train, Y_train)

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
if not os.path.isdir("utils_mfcc"):
    os.mkdir("utils_mfcc")
pickle.dump(model, open("utils_mfcc/classifier_speaker_aug_13mfcc.model", "wb"))

stop=0