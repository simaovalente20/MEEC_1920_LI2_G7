import pyaudio
import wave
import soundfile
import librosa
import numpy as np
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics


#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result


def load_data(test_size = 0.2):
    x, y = [], []
    empty_files = []
    i = 0
    for base_path in glob.glob("Dataset\speaker_class\*"):
        for file in glob.glob(base_path +"\*.wav"):
            basefile = os.path.basename(file)   # get the base name of the audio file
            base_class = base_path.split("\\")[2]
            keyword = basefile.split("_")[1]
            print(base_class + "  ->  " + keyword)
            # remove empty files (G1)
            sound_file = soundfile.SoundFile(file)
            if len(sound_file.read(dtype='float32')) == 0:
                print("Empty File : " + file)
                empty_files.append(file)
                continue
            features = extract_feature(file,mfcc=True,chroma=True, mel=True)
            x.append(features)
            y.append(base_class)
            print(i)
            i = i + 1
    return train_test_split(np.array(x), y, test_size=test_size,random_state=7)


X_train, X_test, Y_train, Y_test = load_data(test_size=0.25)


#https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if not os.path.isdir("audio_utils"):
    os.mkdir("audio_utils")
pickle.dump(scaler, open('audio_utils/scaler_speaker_2class.bin','wb'))

print("[+] Number of training samples:", X_train.shape[0]) # number of samples in training data
print("[+] Number of testing samples:", X_test.shape[0]) # number of samples in testing data
print("[+] Number of features:", X_train.shape[1]) # number of features used, this is a vector of features extracted using extract_features() function

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',max_iter=500)

print("[*] Training the model...")
#model.fit(X_train,Y_train)

clf = OneVsRestClassifier(model)

clf= clf.fit(X_train, Y_train)

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

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("audio_utils"):
    os.mkdir("audio_utils")
pickle.dump(clf, open("audio_utils/classifier_speaker_2class_OvR.model", "wb"))

stop=0