import librosa.display

import matplotlib.pyplot as plt
import numpy as np
import soundfile
import sounddevice as sd
import os, glob, pickle

'''Example'''
#TODO: https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn
word_command = {"Avancar", "Baixo ", "Centro", "Cima", "Direita", "Esquerda", "Parar", "Recuar"}

''' Feature Extraction Utils'''
def read_sounfile(filename):
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
    return X,sample_rate

def extract_feature(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    stft = np.abs(librosa.stft(X, n_fft=1024,hop_length=512))
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

def extract_feature2(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    centroid = kwargs.get("cent")
    rms = kwargs.get("rms")
    mel = kwargs.get("melspec")
    salience = kwargs.get("selience")
    stft = librosa.stft(X, n_fft=1024,hop_length=512)
    mag = np.abs(stft)
    freqs = librosa.core.fft_frequencies(sample_rate)
    result = np.array([])
    if mfcc:    #Mel-frequency cepstral coefficients (MFCCs)
        mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=16, n_fft=1024,hop_length=512)).reshape((-1)) #temporal averaging
        #result = mfccs.flatten()
        result = np.hstack((result, mfccs))
    if centroid:
        cent = np.array(librosa.feature.rms(librosa.magphase(stft,window=np.ones,center = False))).reshape(-1)
        result = np.hstack((result, cent))
    if rms:
        rmss = np.array(librosa.feature.spectral_centroid(y=X, sr=sample_rate,n_fft=1024,hop_length=512)).reshape(-1)
        result = np.hstack((result, rmss))
    if mel:
        mels = np.array(librosa.feature.melspectrogram(X, sr=sample_rate, n_fft=1024,hop_length=512))
        result = mels.flatten()
    return result

def extract_feature3(X, sample_rate, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    pitch = kwargs.get("pitch")
    cqt = kwargs.get("cqt")
    tonnetz = kwargs.get("tonnetz")
    stft = np.abs(librosa.stft(X, n_fft=1024, hop_length=256))

    freqs = librosa.core.fft_frequencies(sample_rate)
    harms = [1, 2, 3, 4]
    weights = [1.0, 0.5, 0.33, 0.25]

    #freqs = librosa.core.fft_frequencies(sample_rate)
    #trimmed, index = librosa.effects.trim(X, top_db=30, frame_length=1024, hop_length=256)
    #print(librosa.get_duration(X,sample_rate), librosa.get_duration(trimmed,sr=sample_rate))
    result = np.array([])
    if mfcc:                                                           #Mel-frequency cepstral coefficients (MFCCs)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13, n_fft=1024,hop_length=256).T, axis=0) #temporal averaging
        result = np.hstack((result, mfccs))
    if chroma:                                                            # compute chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#temporal averaging
        result = np.hstack((result, chroma))
    if pitch:
        #trimmed, index = librosa.effects.trim(X, top_db=30, frame_length=1024, hop_length=512)
        pitches, magnitudes = librosa.piptrack(X,sr = sample_rate,fmin=50.0,fmax=22050.0,threshold=1,ref=np.mean)
        pitch_track = np.array(extract_max(pitches, pitches.shape))
        #p = np.max((pitches).T,axis = 0)
        result = np.hstack((result, pitch_track))
        #m = np.mean((magnitudes).T, axis=0)
        #result = np.hstack((result, m))
    if cqt:
        cqts = np.mean(librosa.feature.chroma_cqt(X, sr=sample_rate,).T,axis=0)#temporal averaging
        result = np.hstack((result, cqts))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

def load_sound_file(file_name):
    data, sample_rate = soundfile.read(file_name,dtype='float32')
    return data, sample_rate

def extract_max(pitches, shape):
    new_pitches = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
    return new_pitches

def smooth(x,window_len=11,window='hanning'):
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

''' Data augmentation Utils'''
def plot_time_series(data,sr):
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr)
    plt.show()

def aug_add_noise(data):
    # Adding white noise
    wn = np.random.randn(len(data))
    data_noise = data + 0.0002 * wn
    return data_noise

def aug_noise(data):
    tmp = data + 0.001*np.random.normal(0,1,len(data))
    return tmp

def aug_pitch(data, sr, pitch_factor):
    return librosa.effects.pitch_shift(data,sr,pitch_factor)

def aug_speed(data, speed_factor):
    tmp = librosa.effects.time_stretch(data,speed_factor)
    return tmp

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

def aug_shift(data,sr,i):
    return  np.roll(data, int((sr*2) *(i/8)))


'''tempo  =0
empty_files = []
for base_path in glob.glob("Dataset_04_07_2020\Dataset\speaker\G*"):
    print(base_path.split("\\")[2])
    for file in glob.glob(base_path + "\*.wav"):
        basename = os.path.basename(file)   # get the base name of the audio file
        print("Grupo " + base_path)
        keyword = basename.split("_")[1]
        print(keyword)
        #print(base_path.split("\\")[3] + "-" + keyword)
        # remove empty files (G1)
        sound_file = soundfile.SoundFile(file)
        if len(sound_file.read(dtype='float32')) == 0:
            print("Empty File : " + file)
            empty_files.append(file)
            continue
        # Raw wave
        sound_frame, sr = read_sounfile(file)
        #plot_time_series(sound_frame, sr)
        #sd.play(sound_frame, sr)
        print(len(sound_frame))
        if (len(sound_frame)) > tempo:
            tempo = len(sound_frame)
            t = basename

        #sound_frame = librosa.util.pad_center(sound_frame,sr*2)
        sound_frame = librosa.util.fix_length(sound_frame, sr*2)
        plot_time_series(sound_frame, sr)
        sd.play(sound_frame, sr)


        # data_noise = aug_add_noise(sound_frame)
        # plot_time_series(data_noise, sr)
        # sd.play(data_noise, sr)

        # data_faster = aug_speed(sound_frame, 1.1)
        # plot_time_series(data_faster, sr)
        # sd.play(data_faster, sr)
        #
        # data_slower = aug_speed(sound_frame, 0.9)
        # plot_time_series(data_slower, sr)
        # sd.play(data_slower, sr)

        # Time Shift with padding
        frame_shift = aug_shift(sound_frame, sr,1)
        plot_time_series(frame_shift, sr)
        sd.play(frame_shift, sr)

        frame_shift = aug_shift(sound_frame, sr,2)
        plot_time_series(frame_shift, sr)
        sd.play(frame_shift, sr)

        frame_shift = aug_shift(sound_frame, sr,3)
        plot_time_series(frame_shift, sr)
        sd.play(frame_shift, sr)
        stop = 1
        
print(tempo)
print(t)
stop = 1'''




