import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS=5
MAX_PLOT_SIZE = CHUNK * 50
DATA = 0

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

