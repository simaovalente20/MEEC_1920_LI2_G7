
import sys, os, glob, pickle
import pyaudio
import wave
import librosa
import sounddevice
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

RECORD_SECONDS=2
MAX_PLOT_SIZE = CHUNK * 50
DATA = 0

FEED_DURATION = 2
FEED_SAMPLES = int(RATE * FEED_DURATION)

keyword_list={"avancar", "baixo ", "centro", "cima", "direita", "esquerda", "parar", "recuar"}
speaker_list={"G1","G2","G3","G4","G5","G6","G7","G8"}

model_keyword = MLPClassifier()
model_keyword = pickle.load(open("result/mlp_classifier_keyword.model", "rb"))
scaler_keyword = StandardScaler()
scaler_keyword = pickle.load(open("result/scaler_keyword.bin", "rb"))

model_speaker = MLPClassifier()
model_speaker = pickle.load(open("result/mlp_classifier_speaker.model", "rb"))
scaler_speaker = StandardScaler()
scaler_speaker = pickle.load(open("result/scaler_speaker.bin", "rb"))

class Audio:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        pass

    def open(self):
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    def record(self):
        self.frames.append(self.stream.read(CHUNK))
        return self.stream.read(CHUNK)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def save(self,file):
        waveFile = wave.open(file,'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()

    def get_audio_input_stream(self):
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("* recording")
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = self.stream.read(CHUNK)
            self.frames.append(data)
        print("* done recording")

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        self.frames = sounddevice.rec(int(FEED_DURATION * RATE), samplerate=RATE, channels=2)
        sounddevice.wait()
        self.frames = librosa.to_mono(self.frames)

        return self.frames

    def extract_features_keyword(self, X):
        keyword_temp = []
        # X = np.fromstring(in_data,dtype=np.float32)
        stft = np.abs(librosa.stft(X))
        result = np.array([])
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=RATE, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=RATE).T, axis=0)
        result = np.hstack((result, chroma))
        mel = np.mean(librosa.feature.melspectrogram(X, sr=RATE).T, axis=0)
        result = np.hstack((result, mel))
        keyword_temp.append(result)
        keyword_normalized = scaler_keyword.transform(keyword_temp)
        return keyword_normalized

    def extract_features_speaker(self, X):
        # X = np.fromstring(in_data, dtype=np.float32)
        speaker_temp = []
        stft = np.abs(librosa.stft(X))
        result = np.array([])
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=RATE, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=RATE).T, axis=0)
        result = np.hstack((result, chroma))
        mel = np.mean(librosa.feature.melspectrogram(X, sr=RATE).T, axis=0)
        result = np.hstack((result, mel))
        speaker_temp.append(result)
        speaker_normalized = scaler_speaker.transform(speaker_temp)
        return speaker_normalized

    def realtime_predict(self, keyword, speaker):
        keyword_prediction = model_keyword.predict(keyword)
        speaker_prediction = model_speaker.predict(speaker)


        keyword_result = keyword_prediction[0]
        speaker_result = speaker_prediction[0]

        #keyword_result = keyword_list[str(keyword_prediction)]
        #speaker_result = speaker_list[str(speaker_prediction)]

        return keyword_result, speaker_result

