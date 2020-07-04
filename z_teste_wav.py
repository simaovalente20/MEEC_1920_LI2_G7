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

# Keyword Scaler/Model
scaler_keyword = StandardScaler()
scaler_keyword = pickle.load(open("utils_audio/scaler_keyword_aug.bin", "rb"))
#model_keyword = MLPClassifier()
#model_keyword = pickle.load(open("result/mlp_classifier_keyword.model", "rb"))
OvR_model_keyword = OneVsRestClassifier(MLPClassifier())
OvR_model_keyword = pickle.load(open("utils_audio/classifier_keyword_aug.model", "rb"))

# Speaker Scaler/Model
scaler_speaker = StandardScaler()
scaler_speaker = pickle.load(open("utils_audio/scaler_speaker_aug.bin", "rb"))
#model_speaker = MLPClassifier()
#model_speaker = pickle.load(open("result/mlp_classifier_speaker.model", "rb"))
OvR_model_speaker = OneVsRestClassifier(MLPClassifier())
OvR_model_speaker = pickle.load(open("utils_audio/classifier_speaker_OvR_aug.model", "rb"))


#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:  # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40,n_fft=1024).T, axis=0)  # temporal averaging
            result = np.hstack((result, mfccs))
        if chroma:  # compute chroma energy
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)  # temporal averaging
            result = np.hstack((result, chroma))
        if mel:  # Mel-scaled spectrogram
            mel = np.mean(librosa.feature.melspectrogram(X, n_fft=1024, sr=sample_rate).T, axis=0)  # temporal averaging
            result = np.hstack((result, mel))
    return result

def extract_feature_speaker(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:  # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=120, n_fft=1024).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:  # compute chroma energy
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)  # temporal averaging
            result = np.hstack((result, chroma))
        if mel:  # Mel-scaled spectrogram
            mel = np.mean(librosa.feature.melspectrogram(X, n_fft=1024, sr=sample_rate).T, axis=0)  # temporal averaging
            result = np.hstack((result, mel))
    return result

filename = ("teste_file//baixo_noise_1.wav")
#filename = ("teste_file//baixo_teste.wav")
#filename = ("teste_file//G7_Baixo.wav")
#filename = ("teste_file//teste_baixo_8.wav")

sound_file = soundfile.SoundFile(filename)
keyword_features = extract_feature(filename, mfcc=True, chroma=True, mel=True)
speaker_features = extract_feature(filename, mfcc=True, chroma=True, mel=True)

keyword_normalized = scaler_keyword.transform(keyword_features.reshape(1, -1))
speaker_normalized = scaler_speaker.transform(speaker_features.reshape(1, -1))

keyword_prediction = OvR_model_keyword.predict(keyword_normalized)
speaker_prediction = OvR_model_speaker.predict(speaker_normalized)

keyword_result = keyword_prediction[0]
speaker_result = speaker_prediction[0]

print(keyword_result)
print(speaker_result)


