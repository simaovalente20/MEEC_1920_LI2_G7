
import sys, os, glob, pickle,time
import pyaudio
import wave
import librosa
import sounddevice
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import struct
import copy
from collections import deque
import threading
from y_audio_utils import read_sounfile, extract_feature, aug_speed, aug_add_noise, aug_shift_zero

FILENAME = "other_sounds/z_file.wav"

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS= 2

MAX_PLOT_SIZE = CHUNK * 50
QUEUE_TIME = RATE*RECORD_SECONDS
#QUEUE_TIME = (int(RATE/(CHUNK*RECORD_SECONDS)))

# Keyword Scaler/Model
scaler_keyword = StandardScaler()
scaler_keyword = pickle.load(open("audio_utils/scaler_keyword_aug.bin", "rb"))
#model_keyword = MLPClassifier()
#model_keyword = pickle.load(open("result/mlp_classifier_keyword.model", "rb"))
OvR_model_keyword = OneVsRestClassifier(MLPClassifier())
OvR_model_keyword = pickle.load(open("audio_utils/classifier_keyword_2class_OvR_aug.model", "rb"))

OvR_model_keyword_augmented = OneVsRestClassifier(MLPClassifier())
OvR_model_keyword_augmented = pickle.load(open("utils_audio/classifier_keyword_aug.model", "rb"))
scaler_keyword_augmented = StandardScaler()
scaler_keyword_augmented = pickle.load(open("utils_audio/scaler_keyword_aug.bin", "rb"))
OvR_model_speaker_augmented = OneVsRestClassifier(MLPClassifier())
OvR_model_speaker_augmented = pickle.load(open("utils_audio/classifier_speaker_OvR_aug.model", "rb"))
scaler_speaker_augmented = StandardScaler()
scaler_speaker_augmented = pickle.load(open("utils_audio/scaler_speaker_aug.bin", "rb"))

# Speaker Scaler/Model
scaler_speaker = StandardScaler()
scaler_speaker = pickle.load(open("audio_utils/scaler_speaker_aug.bin", "rb"))
#model_speaker = MLPClassifier()
#model_speaker = pickle.load(open("result/mlp_classifier_speaker.model", "rb"))

OvR_model_speaker = OneVsRestClassifier(MLPClassifier())
OvR_model_speaker = pickle.load(open("audio_utils/classifier_speaker_2class_OvR_aug.model", "rb"))


class Audio:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.frame_buffer = []
        self.counter = 0
        self.lock = threading.Lock()
        self.stop = False
        self.d=deque(maxlen=QUEUE_TIME)
        pass

    def open(self):
        #self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=self.new_frame)
        print(self.stream.is_active())

        return self.stream

    def start(self):
        self.stream.start_stream()

    def record(self):
        self.frames.append(self.stream.read(CHUNK))
        return self.stream.read(CHUNK)

    '''def close(self):
        self.stream.stop_stream()
        self.stream.close()
        #self.audio.terminate()'''

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        #self.audio.terminate()

    def save(self,file):
        waveFile = wave.open(file,'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()

    def callbackFunc(self, in_data, frame_count, time_info, status_flags):
        print("Callback...")
        print(len(in_data))
        data = np.fromstring(in_data,np.float32)
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

        '''self.d.append(in_data)
        # If 2s worth of audio is collected, copy to secondary buffer and pass to thread function
        if len(self.d) == QUEUE_TIME:
            frames = copy.copy(self.d)
            thread = threading.Thread(target=self.prd_complete())
            thread.start()
            self.d.clear()
            print("2s Buffer")
        return in_data, pyaudio.paContinue
        '''

    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
        return frames

    def new_frame(self, in_data, frame_count, time_info, flag):
        if flag:
            print("Playback Error: %i" % flag)
        print("Callback...")
        print(len(in_data))
        data = np.fromstring(in_data, np.float32)
        amplitude = np.hstack(data)
        #print(amplitude)
        self.frame_buffer.append(amplitude)
        # If 2s worth of audio is collected, copy to secondary buffer and pass to thread function
        if len(self.frame_buffer) >= 20:
            frames = copy.copy(self.frame_buffer)
            thread = threading.Thread(target=self.prd_complete(arg=[frames]))
            thread.start()
            self.frame_buffer.clear()
            #self.d.clear()
            print("2s Buffer")
        #print(len(self.d))
        #played_frames = self.counter
        #self.counter += frame_count
        #return signal[played_frames:self.counter], pyaudio.paContinue
        return in_data, pyaudio.paContinue

    def prd_complete(self,arg):
        print("*****************************************")
        #print(arg)
        data = np.hstack(arg)
        print(len(data))
        #return data
        #self.extract_features_speaker(clip)
        #self.extract_features_keyword(clip)
        # keyword = mic.extract_features_keyword(sound_clip)
        # speaker = mic.extract_features_speaker(sound_clip)
        # keyword_prd , speaker_prd = mic.realtime_predict(keyword,speaker)


    def get_audio_input_stream(self):
        print("* recording")
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = self.stream.read(CHUNK)
            #self.frames.append(data)
            self.frames.append(np.fromstring(data,dtype='Float16'))
        print(int(RATE / CHUNK * RECORD_SECONDS))
        print("finished recording")

        # Unpack data using struct
        #unpack_data = (struct.unpack('h' * chunk, data))

        amplitude = np.hstack(self.frames)
        self.stream.stop_stream()
        self.stream.close()
        self.save(FILENAME)

        '''print("* recording")
        frames_mono = sounddevice.rec(int(RATE * FEED_DURATION), samplerate=RATE, channels=1,dtype='float32')
        sounddevice.wait()
        print("finished recording")'''

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
        keyword_normalized = scaler_keyword.transform(result.reshape(1,-1))
        #keyword_temp.append(result)
        #keyword_normalized = scaler_keyword.transform(keyword_temp)
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
        speaker_normalized = scaler_speaker.transform(result.reshape(1, -1))
        #speaker_temp.append(result)
        #speaker_normalized = scaler_speaker.transform(speaker_temp)
        return speaker_normalized

    def realtime_predict(self, keyword, speaker):
        keyword_prediction = OvR_model_keyword.predict(keyword)
        speaker_prediction = OvR_model_speaker.predict(speaker)

        keyword_result = keyword_prediction[0]
        speaker_result = speaker_prediction[0]

        #keyword_result = keyword_list[str(keyword_prediction)]
        #speaker_result = speaker_list[str(speaker_prediction)]

        return keyword_result, speaker_result

    def extract_features_keyword_augmented(self, X, sr):
        speaker_normalized=[]

        features = extract_feature(X, sr, mfcc=True, chroma=True, mel=True)
        speaker_normalized.append(features)

        frame_shift = aug_shift_zero(X, sr, 0.2, shift_direction='both')
        features = extract_feature(frame_shift, sr, mfcc=True, chroma=True, mel=True)
        speaker_normalized.append(features)

        frame_slower = aug_speed(X, 0.9)
        features = extract_feature(frame_slower, sr, mfcc=True, chroma=True, mel=True)
        speaker_normalized.append(features)

        frame_faster = aug_speed(X, 1.1)
        features = extract_feature(frame_faster, sr, mfcc=True, chroma=True, mel=True)
        speaker_normalized.append(features)

        return speaker_normalized

    def extract_features_speaker_augmented(self, X, sr):
        speaker_normalized = []

        features = extract_feature(X, sr, mfcc=True, chroma=True, mel=True)
        speaker_normalized.append(features)

        frame_shift = aug_shift_zero(X, sr, 0.2, shift_direction='both')
        features = extract_feature(frame_shift, sr, mfcc=True, chroma=True, mel=True)
        speaker_normalized.append(features)

        return speaker_normalized

    def realtime_predict_augmented(self, keyword, speaker):
        speaker_normalized = scaler_speaker_augmented.transform(speaker)
        keyword_normalized = scaler_keyword_augmented.transform(keyword)
        keyword_prediction = OvR_model_keyword_augmented.predict(keyword_normalized)
        speaker_prediction = OvR_model_speaker_augmented.predict(speaker_normalized)

        keyword_result = keyword_prediction[0]
        speaker_result = speaker_prediction[0]

        return keyword_result, speaker_result