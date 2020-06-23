import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion
import numpy as np
import video
import audio
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot

#TODO threading for performance improvement
#TODO Remove mic close error
#TODO spectogram graph (Qt Combo Box)


# Convert a Mat to a Pixmap
def img2pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

# Cyclic capture image
def grabFrame():
    cap=cam.capture()

    image=cam.classify(cap)

    window.label_videoCam.setPixmap(img2pixmap(image))

# Cyclic capture sound
def recording():
    global mic, total_data, max_hold
    raw_data = mic.record()
    '''PyQtGraph plot'''
    data_sample = np.fromstring(raw_data, dtype=np.int16) #convert raw bytes to interger
    total_data = np.concatenate([total_data, data_sample])

    if len(total_data) > audio.MAX_PLOT_SIZE:
        total_data = total_data[audio.CHUNK:]
    audio_waveform.setData(total_data)

# Starts image capture
def on_cameraON_clicked():
    cam.open(0);
    qtimerFrame.start(50)

# Stops image capture
def on_cameraOFF_clicked():
    qtimerFrame.stop()
    cam.close()

# Starts sound capture
def on_micOn_clicked():
    #mic.open()
    clip = mic.get_audio_input_stream()
    keyword = mic.extract_features_keyword(clip)
    speaker = mic.extract_features_speaker(clip)
    keyword_prd , speaker_prd = mic.realtime_predict(keyword,speaker)
    print(keyword_prd)
    print(speaker_prd)
    qtimerRecord.start()

# Stops sound capture
def on_micOff_clicked():
    qtimerRecord.stop()
    mic.close()
    mic.save("file.wav")

# Creation of the camera
cam = video.Video()
# Creation of the micro
mic = audio.Audio()

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("prototype.ui")

#Signals
window.btn_cameraOn.clicked.connect(on_cameraON_clicked)
window.btn_cameraOff.clicked.connect(on_cameraOFF_clicked)
window.btn_micOn.clicked.connect(on_micOn_clicked)
window.btn_micOff.clicked.connect(on_micOff_clicked)
window.label_videoCam.setScaledContents(True)

# Audio plot time domain waveform
audio_plot = window.plotWidget
pg.PlotWidget.getPlotItem(audio_plot).setTitle("Audio Signal")
pg.PlotWidget.getPlotItem(audio_plot).showGrid(True, True)
pg.PlotWidget.getPlotItem(audio_plot).addLegend()
pg.PlotWidget.setBackground(audio_plot,'w')
audio_waveform = audio_plot.plot(pen=(24, 215, 248), name = "Waveform")
#pg.PlotWidget.setAntialiasing(window.plotWidget,aa=1)
total_data = []

# Image capture timer
qtimerFrame = QTimer()
qtimerFrame.timeout.connect(grabFrame)

# Micro timer
qtimerRecord = QTimer()
qtimerRecord.timeout.connect(recording)

window.show()
app.exec()
