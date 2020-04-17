import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion
import numpy as nd
import video

def img2pixmap(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def grabFrame():
    image = cam.detectFaces(cam.capture())
    window.label_videoCam.setPixmap(img2pixmap(image))

def on_cameraON_clicked():
    qtimerFrame.start(50)

def on_cameraOFF_clicked():
    qtimerFrame.stop()
    cam.close()

# Creation of the camera
cam = video.Video(0)

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("prototype.ui")
window.btn_cameraOn.clicked.connect(on_cameraON_clicked)
window.btn_cameraOff.clicked.connect(on_cameraOFF_clicked)
#window.labelFrameInput.setScaledContents(False)
window.label_videoCam.setScaledContents(True)

qtimerFrame = QTimer()
qtimerFrame.timeout.connect(grabFrame)

window.show()
app.exec()
