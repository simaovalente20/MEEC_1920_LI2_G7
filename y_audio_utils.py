import librosa.display
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile

'''Example'''
#TODO: https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn
word_command = {"Avancar", "Baixo ", "Centro", "Cima", "Direita", "Esquerda", "Parar", "Recuar"}

def read_sounfile(filename):
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
    return X,sample_rate

def extract_feature(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    stft = np.abs(librosa.stft(X, n_fft=1024))
    result = np.array([])
    if mfcc:                                                           #Mel-frequency cepstral coefficients (MFCCs)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=1024).T, axis=0) #temporal averaging
        result = np.hstack((result, mfccs))
    if chroma:                                                            # compute chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#temporal averaging
        result = np.hstack((result, chroma))
    if mel:                                                                 # Mel-scaled spectrogram
        mel = np.mean(librosa.feature.melspectrogram(X,n_fft=1024, sr=sample_rate).T,axis=0)#temporal averaging
        result = np.hstack((result, mel))
    return result

def extract_featurep(X, sample_rate, **kwargs):
    pitch = kwargs.get("pitch")
    stft = np.abs(librosa.stft(X, n_fft=1024,hop_length=256))
    result = np.array([])
    if pitch:
        pitches, magnitudes = librosa.piptrack(X,sample_rate,S=stft,n_fft=1024,hop_length=256,fmin=50.0,fmax=22050.0)
        p = np.mean((pitches),axis = 1)
        result = np.hstack((result, p))
        m = np.mean((magnitudes), axis=1)
        result = np.hstack((result, m))
    return result

def extract_feature2(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    centroid = kwargs.get("centroid")
    mel = kwargs.get("mel")
    stft = np.abs(librosa.stft(X, n_fft=1024))
    result = np.array([])
    if mfcc:                                                           #Mel-frequency cepstral coefficients (MFCCs)
        mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=1024)) #temporal averaging
        result = mfccs.flatten()
        #librosa.util.fix_length(result, )
    if centroid:
        cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate,n_fft=1024,hop_length=256)
    return result

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

def time_shift_left(data):
    wav_roll = np.roll(data,)

def aug_shift_zero(data, sr, shift_max, shift_direction):
    shift = np.random.randint(sr * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

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


