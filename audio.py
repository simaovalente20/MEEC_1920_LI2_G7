import sys, os, glob, pickle,time
import pyaudio
import wave
import librosa
import sounddevice
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.multiclass import OneVsRestClassifier
import struct
import copy
from collections import deque
import threading
from y_audio_utils import read_sounfile, extract_feature, aug_speed, aug_add_noise, aug_shift_zero,extract_feature2,extract_feature3

FILENAME = "other_sounds/z_file.wav"

# to commit
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS= 2

MAX_PLOT_SIZE = CHUNK * 50
QUEUE_TIME = RATE*RECORD_SECONDS
#QUEUE_TIME = (int(RATE/(CHUNK*RECORD_SECONDS)))

# Keyword Scaler/Model
scaler_keyword_augmented = StandardScaler()
scaler_keyword_augmented = pickle.load(open("utils_mfcc/scaler_keyword_aug.bin", "rb"))
OvR_model_keyword_augmented = MLPClassifier()
OvR_model_keyword_augmented = pickle.load(open("utils_mfcc/classifier_keyword_aug.model", "rb"))
# Speaker Scaler/Model
OvR_model_speaker_augmented = MLPClassifier()
OvR_model_speaker_augmented = pickle.load(open("utils_mfcc/classifier_speaker_aug_13mfcc.model", "rb"))
scaler_speaker_augmented = StandardScaler()
scaler_speaker_augmented = pickle.load(open("utils_mfcc/scaler_speaker_aug_13mfcc.bin", "rb"))

class Audio:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.frame_buffer = []
        self.data =[]
        self.counter = 0
        self.lock = threading.Lock()
        self.stop = False
        self.d=deque(maxlen=QUEUE_TIME)
        self.i = 0
        self.counter=0
        self.keyword_prd=""
        self.speaker_prd=""
        pass

    #def get_results(self):
        #return self.speaker_prd, self.keyword_prd

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

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()

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
        '''
        with self.lock:
            frames = self.frames
            self.frames = []
        '''
        return self.frame_buffer,self.speaker_prd, self.keyword_prd

    def new_frame(self, in_data, frame_count, time_info, flag):
        if flag:
            print("Playback Error: %i" % flag)
       # print("Callback...")
        #print(len(in_data))
        data = np.fromstring(in_data, np.float32)
       # print(len(data))
        self.prd_complete(data)
        return in_data, pyaudio.paContinue

    def prd_complete(self,arg):
        self.frame_buffer.append(arg)
        if(len(self.frame_buffer)==86):
            self.frame_buffer.pop(0)
        self.counter=self.counter+1
        if self.counter==43:
            self.counter=0
            #frames = copy.copy(self.frame_buffer)
            self.data = np.hstack(self.frame_buffer)
            if max(self.data) >= 0.1:
                thread_classifier = threading.Thread(target=self.func_classifier, args=[self.data])
                thread_classifier.start()
                thread_classifier.join()
            else:
                print("Speaker Louder")
        '''
        if len(self.frame_buffer) == 86:
            frames = copy.copy(self.frame_buffer)
            self.data = np.hstack(frames)
            #print(self.data)
            #print(len(self.data))
            del self.frame_buffer[0:44]
            #self.thread_class.start()
            #print(max(self.data))
            if max(self.data) >= 0.1:
                thread_classifier = threading.Thread(target = self.func_classifier, args= [self.data])
                thread_classifier.start()
            else:
                print("Speaker Louder")
        '''

    def func_classifier(self,data):
        keyword = self.extract_features_keyword_augmented(data)
        speaker = self.extract_features_speaker_augmented(data)
        self.keyword_prd , self.speaker_prd = self.realtime_predict_augmented(keyword,speaker)
        #keyword = self.extract_features_keyword_augmented(data, 44100)
        #speaker = self.extract_features_speaker_augmented(data, 44100)
        #keyword_prd, speaker_prd = self.realtime_predict_augmented(keyword, speaker)
        #self.keyword_prd=keyword_prd
        #self.speaker_prd=speaker_prd
        #print(keyword_prd)
        #print(speaker_prd)


    def get_audio_input_stream(self):
        print("* recording")
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = self.stream.read(CHUNK)
            #self.frames.append(data)
            self.frames.append(np.fromstring(data,dtype='Float16'))
        print(int(RATE / CHUNK * RECORD_SECONDS))
        print("finished recording")
        # Unpack data using struct
        amplitude = np.hstack(self.frames)
        self.stream.stop_stream()
        self.stream.close()
        self.save(FILENAME)
        '''print("* recording")
        frames_mono = sounddevice.rec(int(RATE * FEED_DURATION), samplerate=RATE, channels=1,dtype='float32')
        sounddevice.wait()
        print("finished recording")'''
        return self.frames

    def extract_features_keyword_augmented(self, X):
        speaker_normalized=[]
        Y = librosa.util.fix_length(X, RATE * 2)
        features = extract_feature2(Y, RATE, mfcc=True)
        speaker_normalized.append(features)
        return speaker_normalized

    def aug_shift(data, sr, i):
        return np.roll(data, int((sr * 2) * (i / 8)))

    def extract_features_speaker_augmented(self, X):
        speaker_normalized = []
        x, y = [], []

        features = extract_feature3(X, RATE,mfcc=True)
        speaker_normalized.append(features)

        return speaker_normalized

    def realtime_predict_augmented(self, keyword, speaker):
        speaker_normalized = scaler_speaker_augmented.transform(speaker)
        keyword_normalized = scaler_keyword_augmented.transform(keyword)

        keyword_prediction = OvR_model_keyword_augmented.predict(keyword_normalized)
        keyword_probability = OvR_model_keyword_augmented.predict_proba(keyword_normalized)
        speaker_prediction = OvR_model_speaker_augmented.predict(speaker_normalized)
        keyword_result = keyword_prediction[0]
        speaker_result = speaker_prediction[0]

        print(keyword_probability)
        if keyword_probability.max() > 0.85:
            print(keyword_result)
        else:
            keyword_result="Repeat Word"
            print("Repeat Word")
        return keyword_result, speaker_result















# Keyword Scaler/Model
'''
scaler_keyword = StandardScaler()
scaler_keyword = pickle.load(open("utils_mfcc/scaler_keyword_robust_aug.bin", "rb"))
model_keyword = MLPClassifier()
model_keyword = pickle.load(open("utils_mfcc/classifier_keyword_aug.model", "rb"))
OvR_model_keyword = OneVsRestClassifier(MLPClassifier())
OvR_model_keyword = pickle.load(open("utils_audio/classifier_keyword_OvR_aug.model", "rb"))
# Speaker Scaler/Model
scaler_speaker = StandardScaler()
scaler_speaker = pickle.load(open("utils_audio2/scaler_speaker_aug.bin", "rb"))
model_speaker = MLPClassifier()
model_speaker = pickle.load(open("result/mlp_classifier_speaker.model", "rb"))
OvR_model_speaker = OneVsRestClassifier(MLPClassifier())
OvR_model_speaker = pickle.load(open("utils_audio/classifier_speaker_OvR_aug.model", "rb"))
'''

'''
class audioThread(threading.Thread):
    def __init__(self,ThreadId):
        threading.Thread.__init__(self)
        self.ThreadId = ThreadId
        self.stopped = False

    def stop(self):
        self.stopped=True

    def run(self):
        while True:
            if self.stopped:
                self.stopped=False
            return
        print("asddasdasdasdasdadasd")
        Audio.func_classifier()
        time.sleep(0.1)
'''


'''    def extract_features_keyword(self, X):
        keyword_temp = []
        # X = np.fromstring(in_data,dtype=np.float32)
        stft = np.abs(librosa.stft(X, n_fft=1024))
        result = np.array([])
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=44100, n_mfcc=40, n_fft=1024).T, axis=0) #temporal averaging
        result = np.hstack((result, mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=44100).T,axis=0)#temporal averaging
        result = np.hstack((result, chroma))
        mel = np.mean(librosa.feature.melspectrogram(X,n_fft=1024, sr=44100).T,axis=0)#temporal averaging
        result = np.hstack((result, mel))
        keyword_normalized = scaler_keyword.transform(result.reshape(1,-1))
        #keyword_temp.append(result)
        #keyword_normalized = scaler_keyword.transform(keyword_temp)
        return keyword_normalized

    def extract_features_speaker(self, X):
        # X = np.fromstring(in_data, dtype=np.float32)
        speaker_temp = []
        stft = np.abs(librosa.stft(X, n_fft=1024))
        result = np.array([])
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=44100, n_mfcc=40, n_fft=1024).T, axis=0)  # temporal averaging
        result = np.hstack((result, mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=44100).T, axis=0)  # temporal averaging
        result = np.hstack((result, chroma))
        mel = np.mean(librosa.feature.melspectrogram(X, n_fft=1024, sr=44100).T, axis=0)  # temporal averaging
        result = np.hstack((result, mel))
        speaker_normalized = scaler_speaker.transform(result.reshape(1, -1))
        #speaker_temp.append(result)
        #speaker_normalized = scaler_speaker.transform(speaker_temp)
        return speaker_normalized

    def realtime_predict(self, keyword, speaker):
        keyword_prediction = OvR_model_keyword.predict(keyword)
        keyword_probability = OvR_model_keyword.predict_proba(keyword)
        speaker_prediction = OvR_model_speaker.predict(speaker)
        speaker_probability = OvR_model_speaker.predict_proba(speaker)

        print(keyword_probability)
        print(speaker_probability)

        keyword_result = keyword_prediction[0]
        speaker_result = speaker_prediction[0]
        #keyword_result = keyword_list[str(keyword_prediction)]
        #speaker_result = speaker_list[str(speaker_prediction)]
        return keyword_result, speaker_result
'''