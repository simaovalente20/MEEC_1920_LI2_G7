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

def plot_spec(x,sr):
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(x, sr=sr, x_axis='time', y_axis='hz')
    # If to pring log of frequencies
    plt.colorbar()

def plot_mfcc(x,sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(x, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def plot_chroma(x,sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(x, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    plt.show()

def plot_melspec(x,sr):
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(x, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',y_axis = 'mel', sr = sr,fmax = 8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    stft = np.abs(librosa.stft(X,n_fft=1024))
    result = np.array([])
    if mfcc: #Mel-frequency cepstral coefficients (MFCCs)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft= 1024).T,axis=0) #temporal averaging
        result = np.hstack((result, mfccs))
    if chroma: # Compute a chromagram - energy (magnitude) spectrum
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#temporal averaging
        result = np.hstack((result, chroma))
    if mel: # Mel-scaled spectrogram
        mel = np.mean(librosa.feature.melspectrogram(X,n_fft=1024, sr=sample_rate).T,axis=0)#temporal averaging
        result = np.hstack((result, mel))
    return result

def extract_feature_2(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    stft = np.abs(librosa.stft(X,n_fft=1024))
    if mfcc:  # Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=1024)  # temporal averaging
        plot_mfcc(mfccs, sample_rate)
    if chroma:  # Compute a chromagram - energy (magnitude) spectrum
        # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#temporal averaging
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)  # temporal averaging
        plot_chroma(chroma, sample_rate)
    if mel:  # Mel-scaled spectrogram
        # mel = np.mean(librosa.feature.melspectrogram(X,n_fft=1024, sr=sample_rate).T,axis=0)#temporal averaging
        mel = librosa.feature.melspectrogram(X, n_fft=1024, sr=sample_rate)  # temporal averaging
        plot_melspec(mel, sample_rate)
    return 0

def load_data(test_size = 0.2):
    x, y = [], []
    empty_files = []
    for base_path in glob.glob("Dataset_04_07_2020\Dataset\speaker\G*"):
        print(base_path.split("\\")[2])
        for file in glob.glob(base_path + "\*.wav"):
            basename = os.path.basename(file)   # get the base name of the audio file
            print("Grupo " + base_path)
            keyword = basename.split("_")[1]
            #print(base_path.split("\\")[3] + "-" + keyword)
            # remove empty files (G1)
            sound_file = soundfile.SoundFile(file)
            if len(sound_file.read(dtype='float32')) == 0:
                print("Empty File : " + file)
                empty_files.append(file)
                continue
            # Raw wave
            sound_frame, sr = read_sounfile(file)
            sd.play(sound_frame, sr)
            features = extract_feature_2(sound_frame,sr,mfcc=True, chroma=True, mel=True)
            print(len(features))
            x.append(features)
            y.append(keyword)
            # Time Shift with padding
            frame_shift = aug_shift_zero(sound_frame,sr,0.2,shift_direction='both')
            sd.play(frame_shift, sr)
            #sd.play(frame_shift, sr)
            features = extract_feature_2(frame_shift, sr, mfcc=True, chroma=True, mel=True)
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
            sd.play(frame_slower, sr)
            features = extract_feature_2(frame_slower, sr, mfcc=True, chroma=True, mel=True)
            print(len(features))
            x.append(features)
            y.append(keyword)
            # Speed Faster
            frame_faster = aug_speed(sound_frame,1.1)
            #sd.play(frame_faster, sr)
            features = extract_feature_2(frame_faster, sr, mfcc=True, chroma=True, mel=True)
            print(len(features))
            x.append(features)
            y.append(keyword)
    return train_test_split(np.array(x), y, test_size=test_size, stratify=y,random_state=True)

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

'''
Resample
Y_16k = librosa.resample(X, sample_rate, 16000)
Trim
trimmed, index = librosa.effects.trim(X, top_db=30, frame_length=1024, hop_length=256)
print(librosa.get_duration(X,sample_rate), librosa.get_duration(trimmed,sr=sample_rate))
plot_time_series(trimmed,sample_rate)
sd.play(trimmed, sample_rate)
'''

cr=metrics.classification_report(Y_test,Y_predict)
print("Classification Report:")
print(cr)

stop=0