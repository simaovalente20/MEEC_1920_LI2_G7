# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\claud\PycharmProjects\MEEC_1920_LI2_G7\prototype.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(796, 633)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.plotWidget = PlotWidget(self.centralwidget)
        self.plotWidget.setMinimumSize(QtCore.QSize(0, 150))
        self.plotWidget.setObjectName("plotWidget")
        self.verticalLayout_2.addWidget(self.plotWidget)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_videoCam = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_videoCam.sizePolicy().hasHeightForWidth())
        self.label_videoCam.setSizePolicy(sizePolicy)
        self.label_videoCam.setMinimumSize(QtCore.QSize(300, 300))
        self.label_videoCam.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_videoCam.setFrameShape(QtWidgets.QFrame.Box)
        self.label_videoCam.setObjectName("label_videoCam")
        self.horizontalLayout_2.addWidget(self.label_videoCam)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_6.setTextFormat(QtCore.Qt.AutoText)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.btn_cameraOff = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_cameraOff.sizePolicy().hasHeightForWidth())
        self.btn_cameraOff.setSizePolicy(sizePolicy)
        self.btn_cameraOff.setObjectName("btn_cameraOff")
        self.gridLayout_3.addWidget(self.btn_cameraOff, 0, 1, 1, 1)
        self.btn_cameraOn = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_cameraOn.sizePolicy().hasHeightForWidth())
        self.btn_cameraOn.setSizePolicy(sizePolicy)
        self.btn_cameraOn.setObjectName("btn_cameraOn")
        self.gridLayout_3.addWidget(self.btn_cameraOn, 0, 0, 1, 1)
        self.btn_screenshot = QtWidgets.QPushButton(self.centralwidget)
        self.btn_screenshot.setObjectName("btn_screenshot")
        self.gridLayout_3.addWidget(self.btn_screenshot, 1, 0, 1, 2)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.tableWidget_imageFeatures = QtWidgets.QTableWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget_imageFeatures.sizePolicy().hasHeightForWidth())
        self.tableWidget_imageFeatures.setSizePolicy(sizePolicy)
        self.tableWidget_imageFeatures.setObjectName("tableWidget_imageFeatures")
        self.tableWidget_imageFeatures.setColumnCount(2)
        self.tableWidget_imageFeatures.setRowCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_imageFeatures.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_imageFeatures.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_imageFeatures.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_imageFeatures.setHorizontalHeaderItem(1, item)
        self.tableWidget_imageFeatures.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_imageFeatures.verticalHeader().setStretchLastSection(False)
        self.verticalLayout.addWidget(self.tableWidget_imageFeatures)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.formLayout.setContentsMargins(1, 1, 1, 1)
        self.formLayout.setSpacing(10)
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_lastSpeakerId = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_lastSpeakerId.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_lastSpeakerId.sizePolicy().hasHeightForWidth())
        self.lineEdit_lastSpeakerId.setSizePolicy(sizePolicy)
        self.lineEdit_lastSpeakerId.setInputMask("")
        self.lineEdit_lastSpeakerId.setObjectName("lineEdit_lastSpeakerId")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_lastSpeakerId)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEdit_lastKeyword = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_lastKeyword.sizePolicy().hasHeightForWidth())
        self.lineEdit_lastKeyword.setSizePolicy(sizePolicy)
        self.lineEdit_lastKeyword.setObjectName("lineEdit_lastKeyword")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_lastKeyword)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.comboBox_soundGraph = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_soundGraph.setObjectName("comboBox_soundGraph")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_soundGraph)
        self.verticalLayout.addLayout(self.formLayout)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.btn_micOn = QtWidgets.QPushButton(self.centralwidget)
        self.btn_micOn.setObjectName("btn_micOn")
        self.gridLayout_2.addWidget(self.btn_micOn, 0, 0, 1, 1)
        self.btn_micOff = QtWidgets.QPushButton(self.centralwidget)
        self.btn_micOff.setObjectName("btn_micOff")
        self.gridLayout_2.addWidget(self.btn_micOff, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 796, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_videoCam.setText(_translate("MainWindow", "Video"))
        self.label_6.setText(_translate("MainWindow", "Camera Controls:"))
        self.btn_cameraOff.setText(_translate("MainWindow", "Camera Off"))
        self.btn_cameraOn.setText(_translate("MainWindow", "Camera On"))
        self.btn_screenshot.setText(_translate("MainWindow", "ScreenShot"))
        item = self.tableWidget_imageFeatures.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "F1"))
        item = self.tableWidget_imageFeatures.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "F2"))
        item = self.tableWidget_imageFeatures.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Feature"))
        item = self.tableWidget_imageFeatures.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Value"))
        self.label_7.setText(_translate("MainWindow", "Microfone Controls:"))
        self.label_3.setText(_translate("MainWindow", "Last Speaker:"))
        self.label_4.setText(_translate("MainWindow", "Last Keyword:"))
        self.label_5.setText(_translate("MainWindow", "Sound Graph"))
        self.btn_micOn.setText(_translate("MainWindow", "Mic On"))
        self.btn_micOff.setText(_translate("MainWindow", "Mic Off"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())