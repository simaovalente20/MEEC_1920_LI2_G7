import soundfile
import librosa
import numpy as np
import matplotlib as plt
import noisereduce as nr
# Load audio file
audio_data, sampling_rate = librosa.load("C:/Users/claud/PycharmProjects/MEEC_1920_LI2_G7/Dataset/speaker/G7/G7_Avancar_0.wav")
# Noise reduction
noisy_part = audio_data[0:25000]
reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
# Visualize
print("Original audio file:")
plt(audio_data)
print("Noise removed audio file:")
plt(reduced_noise)