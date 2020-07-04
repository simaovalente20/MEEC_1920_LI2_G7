import pyaudio
import wave
import soundfile
import librosa
import librosa.display
import sounddevice as sd
import numpy as np
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from y_audio_utils import aug_pitch,aug_add_noise,aug_speed,plot_time_series,aug_shift_zero
import matplotlib.pyplot as plt

def read_sounfile(filename):
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
    # trimmed, index = librosa.effects.trim(X, top_db=30, frame_length=1024, hop_length=256)
    #print(librosa.get_duration(X, sample_rate), librosa.get_duration(trimmed, sr=sample_rate))
    # plot_time_series(trimmed,sample_rate)
    #sd.play(trimmed, sample_rate)
    return X,sample_rate

'''
Resample
Y_16k = librosa.resample(X, sample_rate, 16000)
Trim
trimmed, index = librosa.effects.trim(X, top_db=30, frame_length=1024, hop_length=256)
print(librosa.get_duration(X,sample_rate), librosa.get_duration(trimmed,sr=sample_rate))
plot_time_series(trimmed,sample_rate)
sd.play(trimmed, sample_rate)
'''
#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    stft = np.abs(librosa.stft(X,n_fft=1024))
    result = np.array([])
    if mfcc: #Mel-frequency cepstral coefficients (MFCCs)
        mfccs = np.stack(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=12, n_fft= 1024).T, axis=0) #temporal averaging
        result = np.hstack((result, mfccs))
    if chroma: # Compute a chromagram from a waveform or power spectrogram.
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#temporal averaging
        result = np.hstack((result, chroma))
    if mel: # Mel-scaled spectrogram
        mel = np.mean(librosa.feature.melspectrogram(X,n_fft=1024, sr=sample_rate).T,axis=0)#temporal averaging
        result = np.hstack((result, mel))
    return result

def load_data(test_size = 0.2):
    x, y = [], []
    empty_files = []
    i = 0
    for base_path in glob.glob("Dataset\keyword_class\*"):
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
            print(basefile)
            # Raw wave
            sound_frame, sr = read_sounfile(file)
            print("Raw Wave")
            features = extract_feature(sound_frame, sr, mfcc=True, chroma=True, mel=True)
            x.append(features)
            y.append(base_class)
            # Shift
            frame_shift = aug_shift_zero(sound_frame,sr,0.2,shift_direction='both')
            print("Shifted Wave")
            plot_time_series(frame_shift, sr)
            sd.play(frame_shift, sr)
            features = extract_feature(frame_shift, sr, mfcc=True, chroma=True, mel=True)
            x.append(features)
            y.append(base_class)
            # Add Noise
                # frame_noise = aug_add_noise(sound_frame)
                # features = extract_feature(frame_noise, sr, mfcc=True, chroma=True, mel=True)
                # x.append(features)
                # y.append(keyword)
            # Pitch
                # frame_pitch = aug_pitch(sound_frame, sr, 0.4)
                # print("Pitch Wave")
                # features = extract_feature(frame_pitch, sr, mfcc=True, chroma=True, mel=True)
                # x.append(features)
                # y.append(base_class)
            # Speed Slower
            frame_slower = aug_speed(sound_frame, 0.9)
            print("Slower Wave")
            features = extract_feature(frame_slower, sr, mfcc=True, chroma=True, mel=True)
            x.append(features)
            y.append(base_class)
            # Speed Faster
            frame_faster = aug_speed(sound_frame, 1.1)
            print("Faster Wave")
            features = extract_feature(frame_faster, sr, mfcc=True, chroma=True, mel=True)
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

stop=0