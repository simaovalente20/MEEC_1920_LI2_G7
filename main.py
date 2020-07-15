import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion
import numpy as np
import video
import audio
import pyqtgraph as pg
import threading
import time


class myThreadVideo (threading.Thread):
   def __init__(self, threadID):
       threading.Thread.__init__(self)
       self.threadID = threadID
       self.stopped=False

   def stop(self):
       self.stopped = True

   def run(self):
       while True:
           if self.stopped:
               self.stopped=False
               cam.close()
               return
           grabFrame()
           time.sleep(0.1)


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
    image = cam.classify(cap)
    window.label_videoCam.setPixmap(img2pixmap(image))

# Cyclic capture sound
def recording():
    global mic, total_data, max_hold
    #raw_data = mic.record()
    #raw_data = mic.get_audio_input_stream()
    raw_data,lastSpeaker,last_keyword = mic.get_frames()
    '''PyQtGraph plot'''
    # data_sample = np.fromstring(raw_data, dtype=np.int16) #convert raw bytes to interger
    #total_data = np.concatenate([total_data, data_sample])
    if len(raw_data) > 0:
        amplitude = np.hstack(raw_data)
        if len(amplitude) > audio.MAX_PLOT_SIZE:
            amplitude = amplitude[audio.CHUNK:]
            window.lineEdit_lastSpeakerId.setText(lastSpeaker)
            window.lineEdit_lastKeyword.setText(last_keyword)
        audio_waveform.setData(amplitude)
    #lastSpeaker,last_keyword=mic.get_results()
    #window.lineEdit_lastSpeakerId.setText(lastSpeaker)
    #window.lineEdit_lastKeyword.setText(last_keyword)

# Starts image capture
def on_cameraON_clicked():
    window.btn_cameraOn.setEnabled(False)
    window.btn_cameraOff.setEnabled(True)
    window.btn_screenshot.setEnabled(True)
    cam.open(0);
    global thread1
    thread1= myThreadVideo(1)
    thread1.start()

# Stops image capture
def on_cameraOFF_clicked():
    window.btn_cameraOff.setEnabled(False)
    window.btn_cameraOn.setEnabled(True)
    window.btn_screenshot.setEnabled(False)
    thread1.stop()

# Starts sound capture
def on_micOn_clicked():
    window.btn_micOn.setEnabled(False)
    window.btn_micOff.setEnabled(True)
    mic.open()
    qtimerRecord.start()

# Stops sound capture
def on_micOff_clicked():
    window.btn_micOn.setEnabled(True)
    window.btn_micOff.setEnabled(False)
    qtimerRecord.stop()
    mic.close()
    mic.save("file.wav")

def on_screenshot():
    cam.screenshot()

def closeEvent():
    print("close")


# Creation of the camera
cam = video.Video()
# Creation of the micro
mic = audio.Audio()

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("prototype.ui")

#Signals
window.btn_cameraOn.clicked.connect(on_cameraON_clicked)
window.btn_cameraOn.setEnabled(True)
window.btn_cameraOff.clicked.connect(on_cameraOFF_clicked)
window.btn_cameraOff.setEnabled(False)
window.btn_micOn.clicked.connect(on_micOn_clicked)
window.btn_micOn.setEnabled(True)
window.btn_micOff.clicked.connect(on_micOff_clicked)
window.btn_micOff.setEnabled(False)
window.label_videoCam.setScaledContents(True)
window.btn_screenshot.clicked.connect(on_screenshot)
window.btn_screenshot.setEnabled(False)

# Audio plot time domain waveform
audio_plot = window.plotWidget
pg.PlotWidget.getPlotItem(audio_plot).setTitle("Audio Signal")
pg.PlotWidget.getPlotItem(audio_plot).showGrid(True, True)
pg.PlotWidget.getPlotItem(audio_plot).addLegend()
pg.PlotWidget.setBackground(audio_plot,'w')
audio_waveform = audio_plot.plot(pen=(24, 215, 248), name = "Waveform")
total_data = []

# Micro timer
qtimerRecord = QTimer()
qtimerRecord.timeout.connect(recording)

window.show()
app.exec()
