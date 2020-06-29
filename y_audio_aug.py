import pyaudio
import wave
import sounddevice as sd
import soundfile
import librosa
import librosa.display
import itertools
import numpy as np
import random
import os, glob, pickle
import matplotlib.pyplot as plt

def load_sound_file(file_name):
    data, sample_rate = soundfile.read(file_name,dtype='float32')

    return data, sample_rate

def plot_time_series(data,sr):
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr)
    plt.show()

def aug_add_noise(data):
    # Adding white noise
    wn = np.random.randn(len(data))
    data_noise = data + 0.001 * wn
    return data_noise

def aug_pitch(data, sr, pitch_factor):
    return librosa.effects.pitch_shift(data,sr,pitch_factor)

def aug_speed(data, speed_factor):
    tmp = librosa.effects.time_stretch(data,speed_factor)
    return tmp
'''
data, fs = load_sound_file("Dataset/keyword_class/baixo/G7_Baixo_3.wav")
plot_time_series(data,fs)
sd.play(data,fs)

data_noise = aug_add_noise(data)
plot_time_series(data_noise,fs)
sd.play(data_noise,fs)

data_pitch = aug_pitch(data,fs,1.2)
plot_time_series(data_pitch,fs)
sd.play(data_pitch,fs)

data_faster = aug_speed(data,1.1)
plot_time_series(data_faster,fs)
sd.play(data_faster,fs)

data_slower = aug_speed(data,0.9)
plot_time_series(data_slower,fs)
sd.play(data_slower,fs)

stop = 1
'''


