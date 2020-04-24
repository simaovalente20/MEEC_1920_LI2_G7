
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import numpy as np
import time
import video
import audio

class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("prototype.ui", self)
        # Creation of the camera
        self.cam = video.Video(0)
        # Creation of the micro
        self.mic = audio.Audio()

        # Image capture timer
        self.qtimerFrame = QTimer()
        self.qtimerFrame.timeout.connect(self.grabFrame())
        # Micro timer
        self.qtimerRecord = QTimer()
        self.qtimerRecord.timeout.connect(self.recording())

        #Signals
        self.btn_cameraOn.clicked.connect(self.on_cameraON_clicked())
        self.btn_cameraOff.clicked.connect(self.on_cameraOFF_clicked())
        self.btn_micOn.clicked.connect(self.on_micOn_clicked())
        self.btn_micOff.clicked.connect(self.on_micOff_clicked())
        self.label_videoCam.setScaledContents(True)

    # Convert a Mat to a Pixmap
    def img2pixmap(self,image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    # Cyclic capture image
    def grabFrame(self):
        self.image = self.cam.detectFaces(self.cam.capture())
        self.label_videoCam.setPixmap(self.img2pixmap(self.image))

    # Cyclic capture sound
    def recording(self):
        self.mic.record(self)

    # Starts image capture
    def on_cameraON_clicked(self):
        self.qtimerFrame.start(50)

    # Stops image capture
    def on_cameraOFF_clicked(self):
        self.qtimerFrame.stop()
        self.cam.close()

    # Starts sound capture
    def on_micOn_clicked(self):
        self.mic.open()
        self.qtimerRecord.start()

    # Stops sound capture
    def on_micOff_clicked(self):
        self.qtimerRecord.stop()
        self.mic.close()
        self.mic.save("file.wav")

"""
#Audio Graph Visualization
traces = dict()
phase = 0
t = np.arange(0, 3.0, 0.01)
pg.PlotWidget.setBackground(window.plotWidget,'w')
#pg.PlotWidget.setXRange(window.plotWidget,(0,audio.MAX_PLOT_SIZE))
window.plotWidget.plot([1,2,3,4,5,6,7,8,9,10], [30,32,34,32,33,31,29,32,35,45])
"""
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainUi()
    window.show()
    sys.exit(app.exec_())
