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


#keyword_list={"avancar", "baixo ", "centro", "cima", "direita", "esquerda", "parar", "recuar"}
#speaker_list={"G1","G2","G3","G4","G5","G6","G7","G8"}

scaler_keyword = StandardScaler()
scaler_keyword = pickle.load(open("result/scaler_keyword.bin", "rb"))
model_keyword = MLPClassifier()
model_keyword = pickle.load(open("result/mlp_classifier_keyword.model", "rb"))
OvR_model_keyword = OneVsRestClassifier(MLPClassifier())
OvR_model_keyword = pickle.load(open("result/new_OvR_mlp_classifier_keyword.model", "rb"))

scaler_speaker = StandardScaler()
scaler_speaker = pickle.load(open("result/scaler_speaker.bin", "rb"))
model_speaker = MLPClassifier()
model_speaker = pickle.load(open("result/mlp_classifier_speaker.model", "rb"))
OvR_model_speaker = OneVsRestClassifier(MLPClassifier())
OvR_model_speaker = pickle.load(open("result/new_OvR_mlp_classifier_speaker.model", "rb"))




#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, **kwargs):
    """
        Extract feature from audio file `file_name`
            Features supported:
                - MFCC (mfcc)
                - Chroma (chroma)
                - MEL Spectrogram Frequency (mel)
                - Contrast (contrast)
                - Tonnetz (tonnetz)
            e.g:
            `features = extract_feature(path, mel=True, mfcc=True)`
    """
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


#filename = ("teste_file//teste_baixo_8.wav")
#filename = ("teste_file//G7_Esquerda.wav")

filename = ("teste_file//frase_clip.wav")

sound_file = soundfile.SoundFile(filename)

features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1,-1)

keyword_normalized = scaler_keyword.transform(features)
speaker_normalized = scaler_speaker.transform(features)

keyword_prediction = OvR_model_keyword.predict(keyword_normalized)
speaker_prediction = OvR_model_speaker.predict(speaker_normalized)

keyword_result = keyword_prediction[0]
speaker_result = speaker_prediction[0]

print(keyword_result)
print(speaker_result)


