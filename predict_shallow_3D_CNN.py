#/************** HEADER *******************/
import numpy as np
import os,cv2
import winreg as _winreg
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage
import tensorflow as tf
from Crypto.Cipher import AES
import base64
from Crypto import Random
import datetime
import calendar
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

#region GUI
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 768)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.grpResult = QtWidgets.QGroupBox(self.centralwidget)
        self.grpResult.setGeometry(QtCore.QRect(20, 390, 981, 301))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setBold(True)
        font.setWeight(75)
        self.grpResult.setFont(font)
        self.grpResult.setObjectName("grpResult")
        self.lblResult = QtWidgets.QLabel(self.grpResult)
        self.lblResult.setGeometry(QtCore.QRect(20, 110, 941, 91))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.lblResult.setFont(font)
        
        self.lblResult.setText("")
        self.lblResult.setWordWrap(True)
        self.lblResult.setObjectName("lblResult")
        self.btnSelectFolder = QtWidgets.QPushButton(self.centralwidget)
        self.btnSelectFolder.setGeometry(QtCore.QRect(820, 250, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnSelectFolder.setFont(font)
        self.btnSelectFolder.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btnSelectFolder.setObjectName("btnSelectFolder")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(20, 250, 791, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setBold(True)
        font.setWeight(75)
        font.setPointSize(12)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        self.btnPredict = QtWidgets.QPushButton(self.centralwidget)
        self.btnPredict.setGeometry(QtCore.QRect(370, 320, 201, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnPredict.setFont(font)
        self.btnPredict.setObjectName("btnPredict")
        self.lblResult_2 = QtWidgets.QLabel(self.centralwidget)
        self.lblResult_2.setGeometry(QtCore.QRect(20, 10, 981, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.lblResult_2.setFont(font)
        self.lblResult_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lblResult_2.setWordWrap(False)
        self.lblResult_2.setObjectName("lblResult_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(55)
        self.statusbar.setFont(font)
        self.rbSah = QtWidgets.QRadioButton(self.centralwidget)
        self.rbSah.setGeometry(QtCore.QRect(20, 120, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rbSah.setFont(font)
        self.rbSah.setChecked(True)
        self.rbSah.setObjectName("rbSah")
        self.rbIch = QtWidgets.QRadioButton(self.centralwidget)
        self.rbIch.setGeometry(QtCore.QRect(200, 120, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rbIch.setFont(font)
        self.rbIch.setObjectName("rbIch")
        self.rbTb = QtWidgets.QRadioButton(self.centralwidget)
        self.rbTb.setGeometry(QtCore.QRect(530, 120, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rbTb.setFont(font)
        self.rbTb.setObjectName("rbTb")
        self.rbTA = QtWidgets.QRadioButton(self.centralwidget)
        self.rbTA.setGeometry(QtCore.QRect(370, 120, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rbTA.setFont(font)
        self.rbTA.setObjectName("rbTA")
        self.rbAll = QtWidgets.QRadioButton(self.centralwidget)
        self.rbAll.setGeometry(QtCore.QRect(690, 120, 331, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.rbAll.setFont(font)
        self.rbAll.setObjectName("rbAll")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        isLicenseValid = ValidateLicense()
        if isLicenseValid == False:
            self.lblResult.setText("License Expired, Please contact administrator")
            self.btnSelectFolder.setEnabled(False)
            self.btnPredict.setEnabled(False)
        else:
            self.lblResult.setText("")
            self.btnSelectFolder.setEnabled(True)
            self.btnPredict.setEnabled(True)
            self.btnPredict.clicked.connect(self.predict)
            self.btnSelectFolder.clicked.connect(self.SelectFolder)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Predict"))
        self.grpResult.setTitle(_translate("MainWindow", "Result"))
        self.btnSelectFolder.setText(_translate("MainWindow", "Folder Path"))
        self.btnPredict.setText(_translate("MainWindow", "Predict"))
        self.lblResult_2.setText(_translate("MainWindow", "PREDICT"))
        self.statusbar.showMessage("Ready", 0)
        self.rbSah.setText(_translate("MainWindow", "NORMAL-SAH"))
        self.rbIch.setText(_translate("MainWindow", "NORMAL-ICH"))
        self.rbTb.setText(_translate("MainWindow", "NORMAL-TB"))
        self.rbTA.setText(_translate("MainWindow", "NORMAL-TA"))
        self.rbAll.setText(_translate("MainWindow", "NORMAL-SAH-ICH-TRAUMA"))

    def predict(self):
        self.statusbar.showMessage("Processing.......", 0)
        folderPath = self.textEdit.toPlainText()
        folderPath = folderPath.replace("\\","/")
        if os.path.isdir(folderPath):
            if not folderPath.endswith("/"):
                folderPath = folderPath + "/"
            filetype = self.get_radiobtnvalue()
            result,ispositive = prepare_data(folderPath,filetype)
            if ispositive == False:
                self.lblResult.setStyleSheet("font: 75 24pt \"Times New Roman\";\n""color:rgb(255, 0, 0)")
                self.lblResult.setText(result)
            else:
                self.lblResult.setStyleSheet("font: 75 24pt \"Times New Roman\";\n""color:rgb(0, 170, 0)")
                self.lblResult.setText(result)
        else:
            print("Invalid file path")
        self.statusbar.showMessage("Completed", 0)

    def get_radiobtnvalue(self):
        if self.rbSah.isChecked() == True:
            return 1
        elif self.rbIch.isChecked() == True:
            return 2
        elif self.rbTA.isChecked() == True:
            return 3
        elif self.rbTb.isChecked() == True:
            return 4
        elif self.rbAll.isChecked() == True:
            return 5
        else:
            return 0

    def SelectFolder(self):
        folderPath = str(QFileDialog.getExistingDirectory(None, "Select Directory",""))
        self.textEdit.setText(folderPath)
#endregion

#region Data Process
def process_data(patient,ImageSize,data_dir):
    path = data_dir + patient
    slices = [cv2.imread(path + '/' + s,0) for s in os.listdir(path)]
    slices = [cv2.resize(np.array(each_slice),(ImageSize,ImageSize)) for each_slice in slices]
 
    np.array(slices)
    f=np.array(np.stack((slices)))
    s=len(f)

    f=scipy.ndimage.interpolation.zoom(f, (28/s,1,1), order=2, mode='nearest')
    
    new_slices = []
    new_slices.append(f)

    return new_slices
def prepare_data(path,filetype):
    data_dir = path
    patients = os.listdir(data_dir)
    ImageSize= 50
    NewData = []

    for num,patient in enumerate(patients): # Vald Data
        data =process_data(patient,ImageSize,data_dir)

    if filetype == 1:
        result = predict_data_SAH(data)
    elif filetype == 2:
        result = predict_data_ICH(data)
    elif filetype == 3:
        result = predict_data_TA(data)
    elif filetype == 4:
        result = predict_data_TB(data)
    elif filetype == 5:
        result = predict_data_ALL(data)
    return result;
#endregion


#region Data Prediction Normal -SAH
def predict_data_SAH(processedData):
    IMG_SIZE_PX = 50
    SLICE_COUNT = 28
    n_classes = 2
    batch_size = 1
    import sys
    test_data = processedData
    session = tf.Session()
    tf.reset_default_graph()
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    keep_rate = 0.8
    Input = tf.reshape(x, [-1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    with tf.name_scope('Conv1'):
        Filter1 = tf.get_variable(name='Filter1', shape=[3,3,3,1,32]) 
        Bias1 = tf.get_variable(name='Bias1', shape=[32]) 
        conv1 = tf.nn.conv3d(input = Input, 
                             filter = Filter1, 
                             strides=[1, 1, 1, 1, 1], 
                             padding='SAME')
        Conv1 = tf.nn.relu(conv1 + Bias1)
        #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 2 CONV2 =============================================================

        #######LAYER 2 POOL2 =============================================================
    with tf.name_scope('Pool1'):
        Pool1 = tf.nn.max_pool3d(input = Conv1, 
                                     ksize=[1, 2, 2, 2, 1],
                                     strides=[1, 2, 2, 2, 1], 
                                     padding='SAME')

            #         print("Conv1 = ", Conv1.get_shape())

        ######LAYER 3 CONV3 =============================================================
    with tf.name_scope('conv2'):
        Filter2 = tf.get_variable(name='Filter2', shape=[3,3,3,32,64])
        Bias2 = tf.get_variable(name='Bias2', shape=[64])
        conv2 = tf.nn.conv3d(input = Pool1, 
                                filter = Filter2, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv2 = tf.nn.relu(conv2 + Bias2)  
        #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Pool2'):
        Pool2 = tf.nn.max_pool3d(input = Conv2, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')   
        #######LAYER 4 CONV4 =============================================================
    with tf.name_scope('conv3'):
        Filter3 = tf.get_variable(name='Filter3', shape=[3,3,3,64,128])
        Bias3 = tf.get_variable(name='Bias3', shape=[128]) 
        conv3 = tf.nn.conv3d(input = Pool2, 
                                filter = Filter3, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv3 = tf.nn.relu(conv3 + Bias3) 
    #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 4 POOL4 =============================================================
    with tf.name_scope('Pool3'):
        Pool3 = tf.nn.max_pool3d(input = Conv3, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')
         #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('fc1') as scope:
        FilterFc1 = tf.get_variable(name='FilterFc1', shape=[25088, 1024])
        BiasFc1 = tf.get_variable(name='BiasFc1', shape=[1024])
        FlatFc1 = tf.reshape(Pool3, [-1, 25088])  
        OutFc1 = tf.nn.relu(tf.matmul(FlatFc1, FilterFc1) + BiasFc1)
        DropOutFc1 = tf.nn.dropout(OutFc1, keep_rate)
    # print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Out') as scope:
        FilterFc2 = tf.get_variable(name='FilterFc2', shape=[1024, n_classes])
        BiasFc2 = tf.get_variable(name='BiasFc2', shape=[n_classes])
        prediction = tf.matmul(DropOutFc1, FilterFc2) + BiasFc2    
    #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) 
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost) 
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", cost)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver() 
    y_score = []
    y_true = []          
    Accuracy = []
    acc=[]
    loss=[]
    sol = []
    with tf.Session() as sess:
        saver.restore(sess, "D:/IotaMedtech/Model/NORMAL-SAH/model.ckpt")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        probabilities = tf.nn.softmax(prediction)
        for data in test_data:
            X = data/255
            X[X<.37]=0
            X[X>.8]=0

        probs = probabilities.eval(feed_dict={x: X})
        pred = prediction.eval(feed_dict={x: X})
        if probs[0][1] == 1.0:
            return "Brain Hemorrhage (SAH) Detected...............",False
        else:
            return "CT Scan is cleared............",True
# #endregion

#region Data Prediction Normal -ICH
def predict_data_ICH(processedData):
    IMG_SIZE_PX = 50
    SLICE_COUNT = 28
    n_classes = 2
    batch_size = 1
    import sys
    test_data = processedData
    session = tf.Session()
    tf.reset_default_graph()
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    keep_rate = 0.8
    Input = tf.reshape(x, [-1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    with tf.name_scope('Conv1'):
        Filter1 = tf.get_variable(name='Filter1', shape=[3,3,3,1,32]) 
        Bias1 = tf.get_variable(name='Bias1', shape=[32]) 
        conv1 = tf.nn.conv3d(input = Input, 
                             filter = Filter1, 
                             strides=[1, 1, 1, 1, 1], 
                             padding='SAME')
        Conv1 = tf.nn.relu(conv1 + Bias1)
        #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 2 CONV2 =============================================================

        #######LAYER 2 POOL2 =============================================================
    with tf.name_scope('Pool1'):
        Pool1 = tf.nn.max_pool3d(input = Conv1, 
                                     ksize=[1, 2, 2, 2, 1],
                                     strides=[1, 2, 2, 2, 1], 
                                     padding='SAME')

            #         print("Conv1 = ", Conv1.get_shape())

        ######LAYER 3 CONV3 =============================================================
    with tf.name_scope('conv2'):
        Filter2 = tf.get_variable(name='Filter2', shape=[3,3,3,32,64])
        Bias2 = tf.get_variable(name='Bias2', shape=[64])
        conv2 = tf.nn.conv3d(input = Pool1, 
                                filter = Filter2, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv2 = tf.nn.relu(conv2 + Bias2)  
        #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Pool2'):
        Pool2 = tf.nn.max_pool3d(input = Conv2, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')   
        #######LAYER 4 CONV4 =============================================================
    with tf.name_scope('conv3'):
        Filter3 = tf.get_variable(name='Filter3', shape=[3,3,3,64,128])
        Bias3 = tf.get_variable(name='Bias3', shape=[128]) 
        conv3 = tf.nn.conv3d(input = Pool2, 
                                filter = Filter3, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv3 = tf.nn.relu(conv3 + Bias3) 
    #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 4 POOL4 =============================================================
    with tf.name_scope('Pool3'):
        Pool3 = tf.nn.max_pool3d(input = Conv3, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')
         #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('fc1') as scope:
        FilterFc1 = tf.get_variable(name='FilterFc1', shape=[25088, 1024])
        BiasFc1 = tf.get_variable(name='BiasFc1', shape=[1024])
        FlatFc1 = tf.reshape(Pool3, [-1, 25088])  
        OutFc1 = tf.nn.relu(tf.matmul(FlatFc1, FilterFc1) + BiasFc1)
        DropOutFc1 = tf.nn.dropout(OutFc1, keep_rate)
    # print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Out') as scope:
        FilterFc2 = tf.get_variable(name='FilterFc2', shape=[1024, n_classes])
        BiasFc2 = tf.get_variable(name='BiasFc2', shape=[n_classes])
        prediction = tf.matmul(DropOutFc1, FilterFc2) + BiasFc2    
    #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) 
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost) 
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", cost)

    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    y_score = []
    y_true = []          
    Accuracy = []
    acc=[]
    loss=[]
    sol = []
    with tf.Session() as sess:
        saver.restore(sess, "D:/IotaMedtech/Model/NORMAL-ICH/model.ckpt")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        probabilities = tf.nn.softmax(prediction)
        for data in test_data:
            X = data/255
            X[X<.5]=0
            X[X>.8]=0

        probs = probabilities.eval(feed_dict={x: X})
        pred = prediction.eval(feed_dict={x: X})
        if probs[0][1] == 1.0:
            return "Brain Hemorrhage (ICH) Detected...............",False
        else:
            return "CT Scan is cleared............", True
# #endregion

#region Data Prediction Normal -TA
def predict_data_TA(processedData):
    IMG_SIZE_PX = 50
    SLICE_COUNT = 28
    n_classes = 2
    batch_size = 1
    import sys
    test_data = processedData
    session = tf.Session()
    tf.reset_default_graph()
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    keep_rate = 0.8
    Input = tf.reshape(x, [-1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    with tf.name_scope('Conv1'):
        Filter1 = tf.get_variable(name='Filter1', shape=[3,3,3,1,32]) 
        Bias1 = tf.get_variable(name='Bias1', shape=[32]) 
        conv1 = tf.nn.conv3d(input = Input, 
                             filter = Filter1, 
                             strides=[1, 1, 1, 1, 1], 
                             padding='SAME')
        Conv1 = tf.nn.relu(conv1 + Bias1)
        #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 2 CONV2 =============================================================

        #######LAYER 2 POOL2 =============================================================
    with tf.name_scope('Pool1'):
        Pool1 = tf.nn.max_pool3d(input = Conv1, 
                                     ksize=[1, 2, 2, 2, 1],
                                     strides=[1, 2, 2, 2, 1], 
                                     padding='SAME')

            #         print("Conv1 = ", Conv1.get_shape())

        ######LAYER 3 CONV3 =============================================================
    with tf.name_scope('conv2'):
        Filter2 = tf.get_variable(name='Filter2', shape=[3,3,3,32,64])
        Bias2 = tf.get_variable(name='Bias2', shape=[64])
        conv2 = tf.nn.conv3d(input = Pool1, 
                                filter = Filter2, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv2 = tf.nn.relu(conv2 + Bias2)  
        #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Pool2'):
        Pool2 = tf.nn.max_pool3d(input = Conv2, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')   
        #######LAYER 4 CONV4 =============================================================
    with tf.name_scope('conv3'):
        Filter3 = tf.get_variable(name='Filter3', shape=[3,3,3,64,128])
        Bias3 = tf.get_variable(name='Bias3', shape=[128]) 
        conv3 = tf.nn.conv3d(input = Pool2, 
                                filter = Filter3, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv3 = tf.nn.relu(conv3 + Bias3) 
    #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 4 POOL4 =============================================================
    with tf.name_scope('Pool3'):
        Pool3 = tf.nn.max_pool3d(input = Conv3, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')
         #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('fc1') as scope:
        FilterFc1 = tf.get_variable(name='FilterFc1', shape=[25088, 1024])
        BiasFc1 = tf.get_variable(name='BiasFc1', shape=[1024])
        FlatFc1 = tf.reshape(Pool3, [-1, 25088])  
        OutFc1 = tf.nn.relu(tf.matmul(FlatFc1, FilterFc1) + BiasFc1)
        DropOutFc1 = tf.nn.dropout(OutFc1, keep_rate)
    # print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Out') as scope:
        FilterFc2 = tf.get_variable(name='FilterFc2', shape=[1024, n_classes])
        BiasFc2 = tf.get_variable(name='BiasFc2', shape=[n_classes])
        prediction = tf.matmul(DropOutFc1, FilterFc2) + BiasFc2    
    #     print("Conv1 = ", Conv1.get_shape())
        
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) 
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost) 
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", cost)

    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    y_score = []
    y_true = []          
    Accuracy = []
    acc=[]
    loss=[]
    sol = []
    with tf.Session() as sess:
        #D:\Satya\deployment\deployment\Twoclass\NORMvsTA\Metadata\Layer3264128\Fold1\Threshold\10\/model.ckpt
        saver.restore(sess, "D:/IotaMedtech/Model/NORMAL-TA/model.ckpt")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        probabilities = tf.nn.softmax(prediction)
        for data in test_data:  
            X = data/255
            X[X<.5]=0
            X[X>.8]=0
        probs = probabilities.eval(feed_dict={x: X})
        pred = prediction.eval(feed_dict={x: X})
        if probs[0][1] == 1.0:
            return "Brain Hemorrhage (Trauma A) Detected...............",False
        else:
            return "CT Scan is cleared............",True
# #endregion

#region Data Prediction Normal -TB
def predict_data_TB(processedData):
    IMG_SIZE_PX = 50
    SLICE_COUNT = 28
    n_classes = 2
    batch_size = 1
    import sys
    test_data = processedData
    session = tf.Session()
    tf.reset_default_graph()
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    keep_rate = 0.8
    Input = tf.reshape(x, [-1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    with tf.name_scope('Conv1'):
        Filter1 = tf.get_variable(name='Filter1', shape=[3,3,3,1,32]) 
        Bias1 = tf.get_variable(name='Bias1', shape=[32]) 
        conv1 = tf.nn.conv3d(input = Input, 
                             filter = Filter1, 
                             strides=[1, 1, 1, 1, 1], 
                             padding='SAME')
        Conv1 = tf.nn.relu(conv1 + Bias1)
        #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 2 CONV2 =============================================================

        #######LAYER 2 POOL2 =============================================================
    with tf.name_scope('Pool1'):
        Pool1 = tf.nn.max_pool3d(input = Conv1, 
                                     ksize=[1, 2, 2, 2, 1],
                                     strides=[1, 2, 2, 2, 1], 
                                     padding='SAME')

            #         print("Conv1 = ", Conv1.get_shape())

        ######LAYER 3 CONV3 =============================================================
    with tf.name_scope('conv2'):
        Filter2 = tf.get_variable(name='Filter2', shape=[3,3,3,32,64])
        Bias2 = tf.get_variable(name='Bias2', shape=[64])
        conv2 = tf.nn.conv3d(input = Pool1, 
                                filter = Filter2, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv2 = tf.nn.relu(conv2 + Bias2)  
        #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Pool2'):
        Pool2 = tf.nn.max_pool3d(input = Conv2, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')   
        #######LAYER 4 CONV4 =============================================================
    with tf.name_scope('conv3'):
        Filter3 = tf.get_variable(name='Filter3', shape=[3,3,3,64,128])
        Bias3 = tf.get_variable(name='Bias3', shape=[128]) 
        conv3 = tf.nn.conv3d(input = Pool2, 
                                filter = Filter3, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv3 = tf.nn.relu(conv3 + Bias3) 
    #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 4 POOL4 =============================================================
    with tf.name_scope('Pool3'):
        Pool3 = tf.nn.max_pool3d(input = Conv3, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')
         #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('fc1') as scope:
        FilterFc1 = tf.get_variable(name='FilterFc1', shape=[25088, 1024])
        BiasFc1 = tf.get_variable(name='BiasFc1', shape=[1024])
        FlatFc1 = tf.reshape(Pool3, [-1, 25088])  
        OutFc1 = tf.nn.relu(tf.matmul(FlatFc1, FilterFc1) + BiasFc1)
        DropOutFc1 = tf.nn.dropout(OutFc1, keep_rate)
    # print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Out') as scope:
        FilterFc2 = tf.get_variable(name='FilterFc2', shape=[1024, n_classes])
        BiasFc2 = tf.get_variable(name='BiasFc2', shape=[n_classes])
        prediction = tf.matmul(DropOutFc1, FilterFc2) + BiasFc2    
    #     print("Conv1 = ", Conv1.get_shape())
        

    with tf.name_scope('Loss'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) 
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost) 
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", cost)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()      
    y_score = []
    y_true = []          
    Accuracy = []
    acc=[]
    loss=[]
    sol = []
    with tf.Session() as sess:
        #D:\Satya\deployment\deployment\Twoclass\NORMvsTB\Metadata\Layer3264128\Fold1\120
        saver.restore(sess, "D:/IotaMedtech/Model/NORMAL-TB/model.ckpt")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        probabilities = tf.nn.softmax(prediction)
        for data in test_data:
            X = data/255
        probs = probabilities.eval(feed_dict={x: X})
        pred = prediction.eval(feed_dict={x: X})
        if probs[0][1] == 1.0:
            return "Brain Hemorrhage (Trauma B) Detected...............",False
        else:
            return "CT Scan is cleared............",True
# #endregion

#region Data Prediction Normal -ALL
def predict_data_ALL(processedData):
    IMG_SIZE_PX = 50
    SLICE_COUNT = 28
    n_classes = 4
    batch_size = 1
    import sys
    test_data = processedData
    session = tf.Session()
    tf.reset_default_graph()
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    keep_rate = 0.8
    Input = tf.reshape(x, [-1, SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1])

    with tf.name_scope('Conv1'):
        Filter1 = tf.get_variable(name='Filter1', shape=[3,3,3,1,32]) 
        Bias1 = tf.get_variable(name='Bias1', shape=[32]) 
        conv1 = tf.nn.conv3d(input = Input, 
                             filter = Filter1, 
                             strides=[1, 1, 1, 1, 1], 
                             padding='SAME')
        Conv1 = tf.nn.relu(conv1 + Bias1)
        #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 2 CONV2 =============================================================

        #######LAYER 2 POOL2 =============================================================
    with tf.name_scope('Pool1'):
        Pool1 = tf.nn.max_pool3d(input = Conv1, 
                                     ksize=[1, 2, 2, 2, 1],
                                     strides=[1, 2, 2, 2, 1], 
                                     padding='SAME')

            #         print("Conv1 = ", Conv1.get_shape())

        ######LAYER 3 CONV3 =============================================================
    with tf.name_scope('conv2'):
        Filter2 = tf.get_variable(name='Filter2', shape=[3,3,3,32,64])
        Bias2 = tf.get_variable(name='Bias2', shape=[64])
        conv2 = tf.nn.conv3d(input = Pool1, 
                                filter = Filter2, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv2 = tf.nn.relu(conv2 + Bias2)  
        #     print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Pool2'):
        Pool2 = tf.nn.max_pool3d(input = Conv2, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')   
        #######LAYER 4 CONV4 =============================================================
    with tf.name_scope('conv3'):
        Filter3 = tf.get_variable(name='Filter3', shape=[3,3,3,64,128])
        Bias3 = tf.get_variable(name='Bias3', shape=[128]) 
        conv3 = tf.nn.conv3d(input = Pool2, 
                                filter = Filter3, 
                                strides=[1, 1, 1, 1, 1], 
                                padding='SAME')
        Conv3 = tf.nn.relu(conv3 + Bias3) 
    #     print("Conv1 = ", Conv1.get_shape())
        #######LAYER 4 POOL4 =============================================================
    with tf.name_scope('Pool3'):
        Pool3 = tf.nn.max_pool3d(input = Conv3, 
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1], 
                             padding='SAME')
         #     print("Conv1 = ", Conv1.get_shape())
        
    with tf.name_scope('fc1') as scope:
        FilterFc1 = tf.get_variable(name='FilterFc1', shape=[25088, 1024])
        BiasFc1 = tf.get_variable(name='BiasFc1', shape=[1024])
        FlatFc1 = tf.reshape(Pool3, [-1, 25088])  
        OutFc1 = tf.nn.relu(tf.matmul(FlatFc1, FilterFc1) + BiasFc1)
        DropOutFc1 = tf.nn.dropout(OutFc1, keep_rate)
    # print("Conv1 = ", Conv1.get_shape())
    with tf.name_scope('Out') as scope:
        FilterFc2 = tf.get_variable(name='FilterFc2', shape=[1024, n_classes])
        BiasFc2 = tf.get_variable(name='BiasFc2', shape=[n_classes])
        prediction = tf.matmul(DropOutFc1, FilterFc2) + BiasFc2    
    #     print("Conv1 = ", Conv1.get_shape())
        
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) ) 
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost) 
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", cost)

    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
             
    y_score = []
    y_true = []          
    Accuracy = []
    acc=[]
    loss=[]
    sol = []
    with tf.Session() as sess:

        saver.restore(sess, "D:/IotaMedtech/Model/NORMAL-SAH-ICH-TRAUMA/model.ckpt")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        probabilities = tf.nn.softmax(prediction)
        for data in test_data:
            X = data/255         
        probs = probabilities.eval(feed_dict={x: X})
        pred = prediction.eval(feed_dict={x: X})
        if probs[0][0] == 1.0:
            return "Brain Hemorrhage (Trauma) Detected...............",False
        elif probs[0][1] == 1.0:
            return "Brain Hemorrhage (SAH) Detected...............",False 
        elif probs[0][2] == 1.0:
            return "Brain Hemorrhage (ICH) Detected...............",False
        else:
            return "CT Scan is cleared............",True
# #endregion

#region Registry Config
REG_PATH = r"SOFTWARE\CodeFarm\Settings"
REG_KEYSTR = r"LicenseKey"
def set_registryvalue(name, value):
    try:
        _winreg.CreateKey(_winreg.HKEY_CURRENT_USER, REG_PATH)
        registry_key = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, REG_PATH, 0, 
                                       _winreg.KEY_WRITE)
        _winreg.SetValueEx(registry_key, name, 0, _winreg.REG_BINARY, value)
        _winreg.CloseKey(registry_key)
        return True
    except WindowsError:
        return False
def get_regsitryvalue(name):
    try:
        registry_key = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, REG_PATH, 0,
                                       _winreg.KEY_READ)
        value, regtype = _winreg.QueryValueEx(registry_key, name)
        _winreg.CloseKey(registry_key)
        return value
    except WindowsError:
        return None
#endregion

#region Encrypt & Decrypt
MASTER_KEY=b'codefarm@1234321'
iv = Random.new().read(AES.block_size)
def decrypt(cipher_text):
    dec_secret = AES.new(MASTER_KEY,AES.MODE_CBC,iv)
    raw_decrypted = dec_secret.decrypt(base64.b64decode(cipher_text))
    return (raw_decrypted[-16:])
def encrypt(inputText):
    
    enc_secret = AES.new(MASTER_KEY,AES.MODE_CBC,iv)
    cipher_text = base64.b64encode(iv + enc_secret.encrypt(inputText))
    return  cipher_text
#endregion

#region Validate License
def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)
def ValidateLicense():
    
    LicenseKeyStr = get_regsitryvalue(REG_KEYSTR)
    currentDate = datetime.datetime.now()
    if LicenseKeyStr == None:
        #/*********** First Time Writing to Registry ***************************/
        LicenseKeyStr = currentDate.strftime("%y%m%d") + " , " + currentDate.strftime("%y%m%d") + " "
        encryptedLicenseStr = encrypt(LicenseKeyStr.encode())
        set_registryvalue(REG_KEYSTR,encryptedLicenseStr)
    else:
        
        #/*********** Decrypt ***************************/
        plaintext = decrypt(LicenseKeyStr).decode("utf-8")
        #print(plaintext)
        #print(currentDate)

        #/*********** Split and Convert to Date ***************************/
        installedDateStr,prevDateStr = plaintext.split(",")
        installedDate = datetime.datetime.strptime(installedDateStr.strip(),"%y%m%d")
        prevDate = datetime.datetime.strptime(prevDateStr.strip(),"%y%m%d")
        expiredDate = add_months(installedDate,6)
        

        #/*********** Date Validation ***************************/
        installedDateStr,prevDateStr = plaintext.split(",")
        if currentDate.date() > expiredDate:
            return False
        elif currentDate.date() < prevDate.date():
            return False
        elif currentDate.date() > prevDate.date():
            LicenseKeyStr = installedDate.strftime("%y%m%d") + " , " + currentDate.strftime("%y%m%d") + " "
            encryptedLicenseStr = encrypt(LicenseKeyStr.encode())
            set_registryvalue(REG_KEYSTR,encryptedLicenseStr)
            return True
        else:
            return True

    return True
#endregion

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

