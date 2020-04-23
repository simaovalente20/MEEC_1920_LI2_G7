import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion
import numpy as np
import video
import audio
import matplotlib.pyplot as plt

# Convert a Mat to a Pixmap
def img2pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

# Cyclic capture image
def grabFrame():
    image, image_gray, rects=cam.detectFaces_dlib(cam.capture())
    image = cam.getLandmaks(image,image_gray,rects)

    window.label_videoCam.setPixmap(img2pixmap(image))

# Cyclic capture sound
def recording():
    mic.record()

# Starts image capture
def on_cameraON_clicked():
    qtimerFrame.start(50)

# Stops image capture
def on_cameraOFF_clicked():
    qtimerFrame.stop()
    cam.close()

# Starts sound capture
def on_micOn_clicked():
    mic.open()
    qtimerRecord.start()

# Stops sound capture
def on_micOff_clicked():
    qtimerRecord.stop()
    mic.close()
    mic.save("file.wav")

# Creation of the camera
cam = video.Video(0)
# Creation of the micro
mic = audio.Audio()

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("prototype.ui")
window.btn_cameraOn.clicked.connect(on_cameraON_clicked)
window.btn_cameraOff.clicked.connect(on_cameraOFF_clicked)
window.btn_micOn.clicked.connect(on_micOn_clicked)
window.btn_micOff.clicked.connect(on_micOff_clicked)
window.label_videoCam.setScaledContents(True)

# Image capture timer
qtimerFrame = QTimer()
qtimerFrame.timeout.connect(grabFrame)

# Micro timer
qtimerRecord = QTimer()
qtimerRecord.timeout.connect(recording)

window.show()
app.exec()
