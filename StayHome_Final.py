
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
import glob
import os
import calendar
import datetime
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize    
from PyQt5 import QtCore, QtWidgets, uic
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


df=pd.read_csv('subway_merge_1.csv', encoding = 'utf-8')
data=pd.read_csv("Cloud&UV08-17 danger25.csv", engine='python', header=0).dropna()
data2=pd.read_csv("highway.csv", engine='python', header=0)
data3=pd.read_csv("THSeoulCrime.csv", engine='python', header=0)


class MyApp(QDialog):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        tabs = QTabWidget()
        
        tabs.addTab(MainTab(), "StayHome-Main")
        tabs.addTab(FirstTab(), '혼잡해 나가지마')
        tabs.addTab(SecondTab(), '위험해 나가지마')
        tabs.addTab(ThirdTab(), '돈나가 나가지마')
        tabs.addTab(FourthTab(), '돈모아 나가지마')

        buttonbox = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttonbox.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(tabs)
        vbox.addWidget(buttonbox)
        
        self.setWindowTitle('StayHome')
        self.setWindowIcon(QIcon('stayhome.png'))

        self.setLayout(vbox)

        self.setWindowTitle('StayHome')
        self.setGeometry(50, 50, 1050, 800)
        
        self.show()
        
        
        
        
class PandasModel(QtCore.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()


class MainTab(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.nameLabel1 = QLabel(self)
        self.nameLabel1.setText('나가기 귀찮은데...')
        self.nameLabel1.setFont(QtGui.QFont("Times New Roman", 16, QtGui.QFont.Bold))
        self.nameLabel1.move(700, 400)
        
        self.nameLabel2 = QLabel(self)
        self.nameLabel2.setText('집에서 쉬고픈데...')
        self.nameLabel2.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.nameLabel2.move(700, 425)
        
        self.nameLabel3 = QLabel(self)
        self.nameLabel3.setText('이사람 싫은데...')
        self.nameLabel3.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.nameLabel3.move(700, 450)
        
        self.nameLabel4 = QLabel(self)
        self.nameLabel4.setText('그런 너를 위해 준비했어!')
        self.nameLabel4.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.nameLabel4.move(700, 475)
        
        self.nameLabel5 = QLabel(self)
        self.nameLabel5.setText('인싸들이여 오라!')
        self.nameLabel5.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.nameLabel5.move(700, 500)
        
        self.nameLabel6 = QLabel(self)
        self.nameLabel6.setText('The Stay Home!!')
        self.nameLabel6.setFont(QtGui.QFont("Century Schoolbook", 23, QtGui.QFont.Bold))
        self.nameLabel6.move(700, 555)
        
        self.nameLabel7 = QLabel(self)
        self.nameLabel7.setText('Produced by. T4IR_JUAN / T4IR_JIN / T4IR_SUP')
        self.nameLabel7.setFont(QtGui.QFont("Century Schoolbook", 8, QtGui.QFont.Bold))
        self.nameLabel7.move(694, 680)



        
        self.labelA = QLabel(self)
        self.labelB = QLabel(self)
        self.labelC = QLabel(self)
        self.labelA.setText('T4IR 클라우드 기반의 빅데이터 분석 전문가 과정 프로젝트')
        self.labelA.setFont(QtGui.QFont("Century Schoolbook", 25, QtGui.QFont.Black))
        self.labelB.setPixmap(QtGui.QPixmap('stayhome.png'))
        self.labelC.setText('Copyright 2019. StayHome all rights reserved / 무단배포금지 ')
        self.labelC.setFont(QtGui.QFont("Century Schoolbook", 9, QtGui.QFont.Bold))
        self.labelA.move(80, 50)
        self.labelB.move(80, 105)
        self.labelC.move(610, 700)
        
        
class FirstTab(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Year
        self.lbl = QLabel('연도', self)
        self.lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.lbl.move(15, 30)

        self.yy = QComboBox(self)
        self.yy.addItem('2019')
        self.yy.addItem('2020')
        self.yy.addItem('2021')
        self.yy.addItem('2022')
        self.yy.addItem('2023')
        self.yy.addItem('2024')
        self.yy.addItem('2025')
        self.yy.addItem('2026')
        self.yy.addItem('2027')
        self.yy.addItem('2028')
        self.yy.addItem('2029')
        self.yy.addItem('2030')
        self.yy.move(55, 28)

#         self.yy.activated[str].connect(self.onActivated)
        
        # Month
        self.lbl = QLabel(' 월', self)
        self.lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.lbl.move(15, 65)

        self.mm = QComboBox(self)
        self.mm.addItem('1')
        self.mm.addItem('2')
        self.mm.addItem('3')
        self.mm.addItem('4')
        self.mm.addItem('5')
        self.mm.addItem('6')
        self.mm.addItem('7')
        self.mm.addItem('8')
        self.mm.addItem('9')
        self.mm.addItem('10')
        self.mm.addItem('11')
        self.mm.addItem('12')
        self.mm.move(55, 63)

#         self.mm.activated[str].connect(self.onActivated)
        
        # Day
        self.lbl = QLabel(' 일', self)
        self.lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.lbl.move(15, 100)

        self.dd = QComboBox(self)
        self.dd.addItem('1')
        self.dd.addItem('2')
        self.dd.addItem('3')
        self.dd.addItem('4')
        self.dd.addItem('5')
        self.dd.addItem('6')
        self.dd.addItem('7')
        self.dd.addItem('8')
        self.dd.addItem('9')
        self.dd.addItem('10')
        self.dd.addItem('11')
        self.dd.addItem('12')
        self.dd.addItem('13')
        self.dd.addItem('14')
        self.dd.addItem('15')
        self.dd.addItem('16')
        self.dd.addItem('17')
        self.dd.addItem('18')
        self.dd.addItem('19')
        self.dd.addItem('20')
        self.dd.addItem('21')
        self.dd.addItem('22')
        self.dd.addItem('23')
        self.dd.addItem('24')
        self.dd.addItem('25')
        self.dd.addItem('26')
        self.dd.addItem('27')
        self.dd.addItem('28')
        self.dd.addItem('29')
        self.dd.addItem('30')
        self.dd.addItem('31')
        self.dd.move(55, 98)
        
        # 버튼 세팅
        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout(self)


        self.btn1 = QPushButton('Ok', self)
        self.btn1.move(300, 350)
        self.btn1.clicked.connect(self.whichbtn)

        self.btn2 = QPushButton('Exit', self)
        self.btn2.move(200, 350)
        self.btn2.clicked.connect(self.close)        
        hLayout.addWidget(self.btn1)
        hLayout.addWidget(self.btn2)

        self.pandasTv = QtWidgets.QTableView(self)
        hLayout.addWidget(self.pandasTv)
        vLayout.addLayout(hLayout)        
        
        self.setWindowTitle('StayHome')
        self.setGeometry(300, 300, 400, 400)
        self.show()
        
        
    # 버튼 눌렀을 때, 출력
    def whichbtn(self):
#         print(self.yy.currentText(), self.mm.currentText(), self.dd.currentText())
        df1 = self.subway(int(self.yy.currentText()), int(self.mm.currentText()), int(self.dd.currentText()))
        model = PandasModel(df1)
        self.pandasTv.setModel(model)
        self.pandasTv.update()  
        #self.pandasTv.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def subway(self, yyy, mmm, ddd):

        date=list(df.iloc[:,0])
        datelist=[]
        datelist2=[]
        for i in range(len(df)):
            date2=str(date[i])
            year=date2[:4]
            month=date2[4:6]
            day=date2[6:8]
            da=datetime.date(int(year),int(month),int(day)).isocalendar()[1:]
            da2=datetime.date(int(year),int(month),int(day)).isocalendar()
            datelist.append(da)
            datelist2.append(da2)
        datelist
        df["주차요일"] = datelist
        df["년주차요일"] = datelist2
        df["혼잡도"] = (df["승차총승객수"] + df["하차총승객수"])//1000
        #df["승하차총승객수"] = df["승차총승객수"] + df["하차총승객수"]
        df.head()

        sums=df["혼잡도"].groupby([df["년주차요일"],df["주차요일"],df["역명"]]).sum()
        sums2=sums.reset_index(name="혼잡도").reindex(columns=["년주차요일","주차요일","역명","혼잡도"])
        date3=pd.DataFrame(sums2)
        means=date3["혼잡도"].groupby([date3["주차요일"],date3["역명"]]).mean()
        means2=means.reset_index(name="혼잡도").reindex(columns=["주차요일","역명","혼잡도"])
        obj=pd.DataFrame(means2)
        obj2=obj.sort_values(by=['주차요일','혼잡도'], ascending=[True, False])
    
        today=datetime.date(int(yyy),int(mmm),int(ddd)).isocalendar()[1:]

        np.where(obj2["주차요일"] == today)
        select_indices = list(np.where(obj2["주차요일"] == today)[0])
        top20=obj2.iloc[select_indices]
        result = top20[['역명','혼잡도']].head(20)
        result1 = result.set_index('역명')
        return result1

class SecondTab(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        # Year
        self.lbl = QLabel('연도', self)
        self.lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.lbl.move(15, 30)

        self.yy = QComboBox(self)
        self.yy.addItem('2019')
        self.yy.addItem('2020')
        self.yy.addItem('2021')
        self.yy.addItem('2022')
        self.yy.addItem('2023')
        self.yy.addItem('2024')
        self.yy.addItem('2025')
        self.yy.addItem('2026')
        self.yy.addItem('2027')
        self.yy.addItem('2028')
        self.yy.addItem('2029')
        self.yy.addItem('2030')
        self.yy.move(55, 28)

#         self.yy.activated[str].connect(self.onActivated)
        
        # Month
        self.lbl = QLabel(' 월', self)
        self.lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.lbl.move(15, 65)

        self.mm = QComboBox(self)
        self.mm.addItem('1')
        self.mm.addItem('2')
        self.mm.addItem('3')
        self.mm.addItem('4')
        self.mm.addItem('5')
        self.mm.addItem('6')
        self.mm.addItem('7')
        self.mm.addItem('8')
        self.mm.addItem('9')
        self.mm.addItem('10')
        self.mm.addItem('11')
        self.mm.addItem('12')
        self.mm.move(55, 63)

#         self.mm.activated[str].connect(self.onActivated)
        
        # Day
        self.lbl = QLabel(' 일', self)
        self.lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.lbl.move(15, 100)

        self.dd = QComboBox(self)
        self.dd.addItem('1')
        self.dd.addItem('2')
        self.dd.addItem('3')
        self.dd.addItem('4')
        self.dd.addItem('5')
        self.dd.addItem('6')
        self.dd.addItem('7')
        self.dd.addItem('8')
        self.dd.addItem('9')
        self.dd.addItem('10')
        self.dd.addItem('11')
        self.dd.addItem('12')
        self.dd.addItem('13')
        self.dd.addItem('14')
        self.dd.addItem('15')
        self.dd.addItem('16')
        self.dd.addItem('17')
        self.dd.addItem('18')
        self.dd.addItem('19')
        self.dd.addItem('20')
        self.dd.addItem('21')
        self.dd.addItem('22')
        self.dd.addItem('23')
        self.dd.addItem('24')
        self.dd.addItem('25')
        self.dd.addItem('26')
        self.dd.addItem('27')
        self.dd.addItem('28')
        self.dd.addItem('29')
        self.dd.addItem('30')
        self.dd.addItem('31')
        self.dd.move(55, 98)
        
        # Hour
        self.lbl = QLabel('시간', self)
        self.lbl.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.lbl.move(15, 135)

        self.hh = QComboBox(self)
        self.hh.addItem('1')
        self.hh.addItem('2')
        self.hh.addItem('3')
        self.hh.addItem('4')
        self.hh.addItem('5')
        self.hh.addItem('6')
        self.hh.addItem('7')
        self.hh.addItem('8')
        self.hh.addItem('9')
        self.hh.addItem('10')
        self.hh.addItem('11')
        self.hh.addItem('12')
        self.hh.addItem('13')
        self.hh.addItem('14')
        self.hh.addItem('15')
        self.hh.addItem('16')
        self.hh.addItem('17')
        self.hh.addItem('18')
        self.hh.addItem('19')
        self.hh.addItem('20')
        self.hh.addItem('21')
        self.hh.addItem('22')
        self.hh.addItem('23')
        self.hh.addItem('24')
        self.hh.move(55, 133)
        
        # 버튼 세팅
        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout(self)


        self.btn1 = QPushButton('Ok', self)
        self.btn1.move(300, 450)
        self.btn1.clicked.connect(self.whichbtn)

        self.btn2 = QPushButton('Exit', self)
        self.btn2.move(200, 450)
        self.btn2.clicked.connect(self.close)        
        hLayout.addWidget(self.btn1)
        hLayout.addWidget(self.btn2)

        self.pandasTv = QtWidgets.QTableView(self)
        hLayout.addWidget(self.pandasTv)
        vLayout.addLayout(hLayout)        
        
        self.setWindowTitle('StayHome')
        self.setGeometry(300, 300, 400, 400)
        self.show()
        
        
    # 버튼 눌렀을 때, 출력
    def whichbtn(self):
#         print(self.yy.currentText(), self.mm.currentText(), self.dd.currentText())
        df1 = self.danger(int(self.yy.currentText()), int(self.mm.currentText()), int(self.dd.currentText()),                         int(self.hh.currentText()))
        model = PandasModel(df1)
        self.pandasTv.setModel(model)
        self.pandasTv.resizeColumnsToContents()
        self.pandasTv.update()  
        #self.pandasTv.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def danger(self, yyy, mmm, ddd, hhh):
        date=list(df.iloc[:,0])

        y=int(self.yy.currentText())
        m=int(self.mm.currentText())
        d=int(self.dd.currentText())
        h=int(self.hh.currentText())

        x=data.iloc[:,:3]
        y=data.iloc[:,-1:].values.reshape(-1,1)
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)

        x1=data.values[:,:2]
        y1=data.values[:,-1:]

        x2_1=data2['time'].values.reshape(-1,1)
        y2_1=data2['hacc.ratio'].values.reshape(-1,1)
        xtrain2_1,xtest2_1,ytrain2_1,ytest2_1=train_test_split(x2_1,y2_1,test_size=0.3,random_state=0)

        x2=data2.iloc[:,:1]
        y2=data2.iloc[:,-1:].values.reshape(-1,1)
        xtrain2,xtest2,ytrain2,ytest2=train_test_split(x2,y2,test_size=0.3,random_state=0)

        x3=data3['Month'].values.reshape(-1,1)
        y3=data3['Crime.occur'].values.reshape(-1,1)
        xtrain3,xtest3,ytrain3,ytest3=train_test_split(x3,y3,test_size=0.3,random_state=0)

        K=3
        KNN=KNeighborsClassifier(K, weights='distance')
        KNN.fit(xtrain,ytrain)
        KResult=KNN.predict(xtest)
        Accuracy=metrics.accuracy_score(ytest,KResult)
        Acc1 = round(Accuracy*100, 2)
        KResult_1=KNN.predict([[m,d,0]])

        KNN.fit(xtrain2,ytrain2)
        KResult2=KNN.predict(xtest2)
        Accuracy2=metrics.accuracy_score(ytest2,KResult2)
        Acc2 = round(Accuracy2*100, 2)
        KResult2_2=KNN.predict([[h]])

        KNN.fit(xtrain3,ytrain3)
        KResult3=KNN.predict(xtest3)
        Accuracy3=metrics.accuracy_score(ytest3,KResult3)
        Acc3 = round(Accuracy3*100, 2)
        KResult3_2=KNN.predict([[m]])
        kkk=KResult_1.tolist()
        lists =[kkk[0], Acc1]
        kkk2=KResult2_2.tolist()
        lists2=[KResult2_2[0], Acc2]
        kkk3=KResult3_2.tolist()
        lists3=[KResult3_2[0], Acc3]
        df_k = pd.DataFrame(lists)
        df_k2 = pd.DataFrame(lists2)
        df_k3 = pd.DataFrame(lists3)
        df_KK = pd.concat([df_k,df_k2],axis=1)
        df_KKK = pd.concat([df_KK,df_k3],axis=1)
        df_KKK=df_KKK.T  
        df_KKK.columns = ['위험도', '정확도(%)']
        df_KKK.index = ["자외선","교통사고","폭력사건"]
        df_KKK.to_csv('위험도.csv', sep=",",encoding='cp949')
        return df_KKK

class ThirdTab(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        


    
    def initUI(self):
        QMainWindow.__init__(self)


        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout(self)
        hLayout2 = QtWidgets.QHBoxLayout(self)
        
        self.nameLabel = QLabel(self)
    
        self.nameLabel.setText('출발지:')
        self.nameLabel.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.line = QLineEdit(self)

        self.line.move(20, 50)
        self.line.resize(100, 32)
        self.nameLabel.move(20, 20)
        
        self.nameLabel2 = QLabel(self)
        self.nameLabel2.setText('도착지:')
        self.nameLabel2.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.line2 = QLineEdit(self)
        
        self.line2.move(20, 110)
        self.line2.resize(100, 32)
        self.nameLabel2.move(20, 80)
        
        
        self.btn1 = QPushButton('확인', self)
        self.btn1.move(300, 200)
        self.btn1.clicked.connect(self.clickMethod)
        
        self.btn2 = QPushButton('결과', self)
        self.btn2.move(300, 200)
        self.btn2.clicked.connect(self.whichbtn)
        
       
        hLayout.addWidget(self.btn1)
        hLayout.addWidget(self.btn2)

        self.pandasTv = QtWidgets.QTableView(self)
        hLayout.addWidget(self.pandasTv)
        vLayout.addLayout(hLayout)   
        
        self.pandasTv2 = QtWidgets.QTableView(self)
        hLayout2.addWidget(self.pandasTv2)
        vLayout.addLayout(hLayout2)     
        

        
    def clickMethod(self):
              
        print('출발지: ' + self.line.text())
        print('도착지: ' + self.line2.text())
        x= self.line.text()
        y = self.line2.text() 
        f = open("C:\python/출발지.txt", 'w',encoding='utf-8')
        f.write(x)
        f.close()
        f = open("C:\python/도착지.txt", 'w',encoding='utf-8')
        f.write(y)
        f.close()
        
        def get_html(url):
            _html = ""
            resp = requests.get(url)
            if resp.status_code == 200:
                _html = resp.text
                return _html
        f1 = open("C:\python/출발지.txt", 'r',encoding='utf-8')
        location = f1.readline()
        f1.close()
        f2 = open("C:\python/도착지.txt", 'r',encoding='utf-8')
        location2 = f2.readline()
        f2.close()
        print("출발지:"+location)
        print("도착지:"+location2)
        import time
        import re
        import urllib
        import sys
        from bs4 import BeautifulSoup as bs
        import requests
        import os
        import pandas as pd
        import itertools 
        import requests
        import pprint
        import json
        now = time.localtime()
        s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
        s2 = "%02d:%02d:%02d"%(now.tm_hour, now.tm_min, now.tm_sec)

        # 요청 주소(구글맵) 
        URL = 'http://maps.googleapis.com/maps/api/geocode/json?sensor=false&language=ko&address={}'.format(location)
        # URL 가져오기
        URL = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyCTr_83ZtolTvfjAkgf_gwsAbDP-bBkwk4&sensor=false&language=ko&address={}'.format(location)
        # URL로 보낸 Requst의 Response를 response 변수에 할당
        response = requests.get(URL)
        # JSON 파싱
        data = response.json()
        # lat, lon 추출
        lat = data['results'][0]['geometry']['location']['lat']
        lng = data['results'][0]['geometry']['location']['lng']
        # print() 함수 대신 pprint.pprint() 함수를 사용하는 이유는 좀 더 보기 쉽게 출력하기 위함입니다.


        # 요청 주소(구글맵) 
        URL2 = 'http://maps.googleapis.com/maps/api/geocode/json?sensor=false&language=ko&address={}'.format(location2)
        # URL 가져오기
        URL2 = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyCTr_83ZtolTvfjAkgf_gwsAbDP-bBkwk4&sensor=false&language=ko&address={}'.format(location2)
        # URL로 보낸 Requst의 Response를 response 변수에 할당
        response2 = requests.get(URL2)
        # JSON 파싱
        data2 = response2.json()
        # lat, lon 추출
        lat2 = data2['results'][0]['geometry']['location']['lat']
        lng2 = data2['results'][0]['geometry']['location']['lng']
        # print() 함수 대신 pprint.pprint() 함수를 사용하는 이유는 좀 더 보기 쉽게 출력하기 위함입니다.



        url = "https://api2.sktelecom.com/tmap/routes/prediction?version=2&totalValue=2"
        data = {
          "routesInfo": {
            "departure": {
              "name": location,
              "lon": lng,
              "lat": lat,
            },
            "destination": {
              "name": location2,
              "lon": lng2,
              "lat": lat2,
            },
            "predictionType": "departure",
            "predictionTime": s+"T"+s2+"+0900",
            "searchOption": "00",
            "tollgateCarType": "car",
            "trafficInfo" : "N"
          }
        }
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain','appKey': '3e23b3af-8311-4b15-9804-cca51d11212d'}
        r = requests.post(url, data=json.dumps(data), headers=headers)
        result = r.content
        string=result.decode('utf-8')
        splits=string.split("  ")
        show=[]
        for i in range(len(splits)):
            if 'totalDistance' in splits[i]:
                show.append(splits[i])
            if 'totalTime' in splits[i]:
                show.append(splits[i])
            if 'totalFare' in splits[i]:
                show.append(splits[i])
            if 'taxiFare' in splits[i]:
                show.append(splits[i])
            if 'departureTime' in splits[i]:
                show.append(splits[i])
            if 'arrivalTime' in splits[i]:
                show.append(splits[i])

        list2=[]
        for j in range(len(show)):
            list2.append(show[j].split(":"))
        times=[]
        for k in [4,5]:
            times.append(show[k].split(" "))
        total=list2[0:3]
        import re
        def cleanText (readData):
            text =re.sub('[,"\n\t]',"",readData)
            return text
        list2=[]
        for j in range(len(show)):
            list2.append(cleanText(show[j]).split(" "))
        times=[]
        for k in [4,5]:
            times.append(cleanText(show[k]).split(" "))
        total=list2[0:3]
        df1 = pd.DataFrame(total)
        df2 = pd.DataFrame(times)
        df2.values[:,1][0]=df2.values[:,1][0][11:19]
        df2.values[:,1][1]=df2.values[:,1][1][11:19]
        df1.values[:,1] = df1.values[:,1].astype('int')
        df3 = pd.concat([df1,df2],axis=0)
        df3.columns = ["구분",'안내 내용']
        Dataframe=df3.set_index('구분')
        Dataframe = Dataframe.rename(index={'totalDistance:':"총거리(Km)",'totalTime:':"총시간(분)",'totalFare:':"톨게이트 비용(원)",'departureTime:':"출발시간",'arrivalTime:':"도착예정시간"})
        Dataframe.values[0,0]=Dataframe.values[0,0]/1000
        Dataframe.values[1,0]=Dataframe.values[1,0]/60

        # 유류정보


        url="https://finance.naver.com/marketindex/?tabSel=gold#tab_section"

        def get_html(url):
            _html = ""
            resp = requests.get(url)
            if resp.status_code == 200:
                _html = resp.text
                return _html

        html = get_html(url)
        soup = bs(html, 'html.parser',from_encoding='utf-8')
        tag = soup.find('table', attrs={'class': 'tbl_exchange'})
        tag2=tag.text
        tag3=cleanText(tag2).split("/")
        tag4=[]
        for ma in range(len(tag3)):
            tag4.append(tag3[ma].split(" "))
        df4 = pd.DataFrame(tag4)
        price=df4.values[1:,1]
        df5 = pd.DataFrame({"휘발유":[price[0]],"고급휘발유":[price[1]], "경유":[price[2]], "두바이유":[price[3]],"브렌트유":[price[4]],"텍사스유":[price[5]]})
        df6=df5.T

        # 기름값
        urls="https://finance.naver.com/marketindex/oilDetail.nhn?marketindexCd=OIL_GSL"
        def get_htmls(urls):
            _html = ""
            resps = requests.get(urls)
            if resps.status_code == 200:
                _htmls = resps.text
                return _htmls

        htmls = get_htmls(urls)
        soups = bs(htmls, 'html.parser',from_encoding='utf-8')
        dd=pd.DataFrame(soups.text.replace("\n"," ").split(" "))
        OIL_GSL_PRICE=float(cleanText(dd.values[17,0]))
        urls2="https://finance.naver.com/marketindex/oilDetail.nhn?marketindexCd=OIL_LO"
        def get_htmls2(urls2):
            _html = ""
            resps2 = requests.get(urls2)
            if resps2.status_code == 200:
                _htmls2 = resps2.text
                return _htmls2

        htmls2 = get_htmls2(urls2)
        soups2 = bs(htmls2, 'html.parser',from_encoding='utf-8')
        dd2=pd.DataFrame(soups2.text.replace("\n"," ").split(" "))
        OIL_LO_PRICE=float(cleanText(dd2.values[17,0]))
        X = Dataframe.values[0,0]
        GSL_Total_price=X/16*OIL_GSL_PRICE
        LO_Total_price=X/14*OIL_LO_PRICE
        price = pd.DataFrame([GSL_Total_price,LO_Total_price], index=["휘발유 기름값(원):","경유 기름값(원):"],columns=['안내 내용'])

        # 표 정리
        df7=pd.concat([Dataframe,price])
        Total_money=(df7.values[2,0]+(df7.values[5,0]+df7.values[6,0])/2)*2
        df8 = pd.DataFrame([Total_money],index=["예상 왕복 비용(원)"],columns=["안내 내용"])
        Final_Chart=pd.concat([df7,df8])
        pd.options.display.float_format='{:.2f}'.format
        table = Final_Chart
        df = table
        import math
        df.iloc[1]=round(float(df.iloc[1]))
        df.iloc[5]=round(float(df.iloc[5]),2)
        df.iloc[6]=round(float(df.iloc[6]),2)
        df.iloc[7]=math.ceil(float(df.iloc[7]))
        df.to_csv("여행비용2.csv", sep=',', encoding='cp949')
        df.to_csv("여행비용.csv", sep=',', encoding='cp949')
        
        
        url = "https://search.shopping.naver.com/best100v2/main.nhn"
        def get_html(url):
            _html = ""
            resp = requests.get(url)
            if resp.status_code == 200:
                _html = resp.text
                return _html

        def cleanText (readData):
            text =re.sub("[,""'\n\t]","",readData)
            return text
        html=get_html(url)
        soup=bs(html,'html.parser',from_encoding='utf-8')
        import re
        img = soup.find_all('img', attrs={'_productLazyImg'})
        num = soup.find_all('span', attrs={'class': 'num'})
        list1 = []
        for i in range(len(img)):
            list1.append(str(img[i]).split('"'))
        list2 = []
        for j in range(len(num)):
            list2.append(cleanText(num[j].text))
        name =[]
        pic_ad =[]
        for k in range(len(list1)):
            name.append(list1[k][1])
            pic_ad.append(list1[k][5])
        name_df = pd.DataFrame(name)
        price_df = pd.DataFrame(list2)
        List = pd.concat([name_df,price_df],axis=1)
        Shop_List=List.drop([0,1,2,3,4]).reset_index(drop=True)
        Shop_List.values[:,1].astype(int)

        clothes=Shop_List[:6]
        grocery = Shop_List[7:13]
        cosmatic=Shop_List[14:20]
        device=Shop_List[21:27]
        furniture=Shop_List[28:34]
        baby=Shop_List[35:41]
        sports=Shop_List[42:48]
        food=Shop_List[49:55]
        health=Shop_List[56:62]

        clothes.columns = ["패션의류 인기상품",'가격']
        grocery.columns = ["패션잡화 인기상품",'가격']
        cosmatic.columns = ["화장품/미용 인기상품",'가격']
        device.columns = ["디지털/가전 인기상품",'가격']
        furniture.columns = ["가구/인테리어 인기상품",'가격']
        baby.columns = ["출산/육아 인기상품",'가격']
        sports.columns = ["스포츠/레저 인기상품",'가격']
        food.columns = ["식품 인기상품",'가격']
        health.columns = ["생활건강 인기상품",'가격']

        clothes.reset_index(drop=True,inplace=True)
        grocery.reset_index(drop=True,inplace=True)
        cosmatic.reset_index(drop=True,inplace=True)
        device.reset_index(drop=True,inplace=True)
        furniture.reset_index(drop=True,inplace=True)
        baby.reset_index(drop=True,inplace=True)
        sports.reset_index(drop=True,inplace=True)
        food.reset_index(drop=True,inplace=True)
        health.reset_index(drop=True,inplace=True)

        clothes=clothes.set_index("패션의류 인기상품")
        grocery = grocery.set_index("패션잡화 인기상품")
        cosmatic = cosmatic.set_index("화장품/미용 인기상품")
        device = device.set_index("디지털/가전 인기상품")
        furniture = furniture.set_index("가구/인테리어 인기상품")
        baby = baby.set_index("출산/육아 인기상품")
        sports =sports.set_index("스포츠/레저 인기상품")
        food = food.set_index("식품 인기상품")
        health = health.set_index("생활건강 인기상품")



        now = time.localtime()
        s = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)
        clothes.to_csv(s+"패션의류 인기상품.csv", sep=',', encoding='cp949')
        grocery.to_csv(s+"패션잡화 인기상품.csv", sep=',', encoding='cp949')
        cosmatic.to_csv(s+"화장품&미용 인기상품.csv", sep=',', encoding='cp949')
        device.to_csv(s+"디지털&가전 인기상품.csv", sep=',', encoding='cp949')
        furniture.to_csv(s+"가구&인테리어 인기상품.csv", sep=',', encoding='cp949')
        baby.to_csv(s+"출산&육아 인기상품.csv", sep=',', encoding='cp949')
        sports.to_csv(s+"스포츠&레저 인기상품.csv", sep=',', encoding='cp949')
        food.to_csv(s+"식품 인기상품.csv", sep=',', encoding='cp949')
        health.to_csv(s+"생활건강 인기상품.csv", sep=',', encoding='cp949')


        data_clothes = pd.read_csv(s+"패션의류 인기상품.csv", encoding='cp949',engine='python')
        data_grocery = pd.read_csv(s+"패션잡화 인기상품.csv", encoding='cp949',engine='python')
        data_cosmetic = pd.read_csv(s+"화장품&미용 인기상품.csv", encoding='cp949',engine='python')
        data_devices = pd.read_csv(s+"디지털&가전 인기상품.csv", encoding='cp949',engine='python')
        data_furniture = pd.read_csv(s+"가구&인테리어 인기상품.csv", encoding='cp949',engine='python')
        data_baby = pd.read_csv(s+"출산&육아 인기상품.csv", encoding='cp949',engine='python')
        data_sport= pd.read_csv(s+"스포츠&레저 인기상품.csv", encoding='cp949',engine='python')
        data_food = pd.read_csv(s+"식품 인기상품.csv", encoding='cp949',engine='python')
        data_health = pd.read_csv(s+"생활건강 인기상품.csv", encoding='cp949',engine='python')
        val_clothes=data_clothes.values[:,1].tolist()
        val_grocery=data_grocery.values[:,1].tolist()
        val_cosmetic=data_cosmetic.values[:,1].tolist()
        val_devices=data_devices.values[:,1].tolist()
        val_furniture=data_furniture.values[:,1].tolist()
        val_baby=data_baby.values[:,1].tolist()
        val_sport=data_sport.values[:,1].tolist()
        val_food=data_food.values[:,1].tolist()
        val_health=data_health.values[:,1].tolist()
        showing_clothes_list=[]
        showing_grocery_list=[]
        showing_cosmetic_list=[]
        showing_device_list=[]
        showing_furniture_list=[]
        showing_baby_list=[]
        showing_sport_list=[]
        showing_food_list=[]
        showing_heath_list=[]
        table = pd.read_csv("여행비용.csv", encoding='cp949',engine='python')
        val=float(table.values[7,1])
        for ai in range(len(val_clothes)):
            if val_clothes[ai] <= val:
                showing_clothes_list.append(val_clothes[ai])
            else:
                showing_grocery_list.append(int(0))
            if val_grocery[ai] <= val:
                showing_grocery_list.append(val_grocery[ai])
            else:
                showing_grocery_list.append(int(0))
            if val_cosmetic[ai] <= val:
                showing_cosmetic_list.append(val_cosmetic[ai])
            else:
                showing_cosmetic_list.append(int(0))
            if val_devices[ai] <= val:
                showing_device_list.append(val_devices[ai])
            else:
                showing_device_list.append(int(0))
            if val_furniture[ai] <= val:
                showing_furniture_list.append(val_furniture[ai])
            else:
                showing_furniture_list.append(int(0))
            if val_baby[ai] <= val:
                showing_baby_list.append(val_baby[ai])
            else:
                showing_baby_list.append(int(0))
            if val_sport[ai] <= val:
                showing_sport_list.append(val_sport[ai])
            else:
                showing_sport_list.append(int(0))
            if val_food[ai] <= val:
                showing_food_list.append(val_food[ai])
            else:
                showing_food_list.append(int(0))
            if val_health[ai] <= val:
                showing_heath_list.append(val_health[ai])
            else:
                showing_heath_list.append(int(0))
        a=[]
        all_clothes=data_clothes.values[:,:].tolist()
        all_grocery=data_grocery.values[:,:].tolist()
        all_cosmetic=data_cosmetic.values[:,:].tolist()
        all_devices=data_devices.values[:,:].tolist()
        all_furniture=data_furniture.values[:,:].tolist()
        all_baby=data_baby.values[:,:].tolist()
        all_sport=data_sport.values[:,:].tolist()
        all_food=data_food.values[:,:].tolist()
        all_health=data_health.values[:,:].tolist()
        list_clothes=[]
        list_grocery=[]
        list_cosmetic=[]
        list_devices=[]
        list_furniture=[]
        list_baby=[]
        list_sport=[]
        list_food=[]
        list_health=[]
        for bi in all_clothes:
            if bi[:][1] in showing_clothes_list:
                list_clothes.append(bi)
            else:
                list_clothes.append(['-','-'])
        for bi in all_grocery:
            if bi[:][1] in showing_grocery_list:
                list_grocery.append(bi)
            else:
                list_grocery.append(['-','-'])
        for bi in all_cosmetic:
            if bi[:][1] in showing_cosmetic_list:
                list_cosmetic.append(bi)
            else:
                list_cosmetic.append(['-','-'])
        for bi in all_devices:
            if bi[:][1] in showing_device_list:
                list_devices.append(bi)
            else:
                list_devices.append(['-','-'])
        for bi in all_furniture:
            if bi[:][1] in showing_furniture_list:
                list_furniture.append(bi)
            else:
                list_furniture.append(['-','-'])
        for bi in all_baby:
            if bi[:][1] in showing_baby_list:
                list_baby.append(bi)
            else:
                list_baby.append(['-','-'])
        for bi in all_sport:
            if bi[:][1] in showing_sport_list:
                list_sport.append(bi)
            else:
                list_sport.append(['-','-'])
        for bi in all_food:
            if bi[:][1] in showing_food_list:
                list_food.append(bi)
            else:
                list_food.append(['-','-'])
        for bi in all_health:
            if bi[:][1] in showing_heath_list:
                list_health.append(bi)
            else:
                list_health.append(['-','-'])
        df_clothes = pd.DataFrame(list_clothes)
        df_grocery = pd.DataFrame(list_grocery)
        df_cosmetic = pd.DataFrame(list_cosmetic)
        df_devices = pd.DataFrame(list_devices)
        df_furniture = pd.DataFrame(list_furniture)
        df_baby = pd.DataFrame(list_baby)
        df_sport = pd.DataFrame(list_sport)
        df_food = pd.DataFrame(list_food)
        df_health = pd.DataFrame(list_health)

        pieces = {'패션의류 인기상품': df_clothes, 
                          '패션잡화 인기상품': df_grocery,
                          "화장품/미용 인기상품": df_cosmetic,
                          "디지털/가전 인기상품": df_devices,
                          "가구/인테리어 인기상품":df_furniture,
                          "출산/육아 인기상품":df_baby,
                          "스포츠/레저 인기상품":df_sport,
                          "식품 인기상품":df_food,
                          "생활건강 인기상품":df_health}
        Final_view=pd.concat(pieces)
        indexNames = Final_view[ Final_view.values[:,1] == '-' ].index
        # Delete these row indexes from dataFrame
        Final_view.drop(indexNames , inplace=True)
        Final_view.to_csv("쇼핑추천리스트.csv", sep=',', encoding='cp949')
        
    
    def whichbtn(self):
        df = pd.read_csv("여행비용.csv", encoding='cp949',engine='python',index_col=[0])
        print(df)
        model = PandasModel(df)
        self.pandasTv.setModel(model)
        self.pandasTv.resizeColumnsToContents()
        self.pandasTv.update()  

        df2 = pd.read_csv("쇼핑추천리스트.csv", encoding='cp949',engine='python',index_col=0)
        df2 = df2.rename(index=str, columns={"0": "상품 리스트", "1": "가격"})
        df2.drop(['Unnamed: 1'], axis=1, inplace=True)
        model2 = PandasModel(df2)
        self.pandasTv2.setModel(model2)
        self.pandasTv2.resizeColumnsToContents()
        self.pandasTv2.update()
    
class FourthTab(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        


    
    def initUI(self):
        QMainWindow.__init__(self)


        vLayout = QtWidgets.QVBoxLayout(self)
        vLayout2 = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout(self)
        hLayout2 = QtWidgets.QHBoxLayout(self)
        
        self.nameLabel = QLabel(self)
    
        self.nameLabel.setText('구매물품:')
        self.nameLabel.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.line = QLineEdit(self)

        self.line.move(20, 120)
        self.line.resize(100, 32)
        self.nameLabel.move(20, 85)
        
        self.nameLabel2 = QLabel(self)
        self.nameLabel2.setText('목표금액:')
        self.nameLabel2.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.line2 = QLineEdit(self)
        
        self.line2.move(20, 475)
        self.line2.resize(100, 32)
        self.nameLabel2.move(20, 440)
        
        
        self.btn1 = QPushButton('확인', self)
        self.btn1.move(300, 200)
        self.btn1.clicked.connect(self.clickMethod)
        
        self.btn2 = QPushButton('결과', self)
        self.btn2.move(300, 200)
        self.btn2.clicked.connect(self.whichbtn)
        
        hLayout.addWidget(self.btn1)
        hLayout.addWidget(self.btn2)
                
        
        self.btn3 = QPushButton('확인', self)
        self.btn3.clicked.connect(self.clickMethod2)
        
        self.btn4 = QPushButton('결과', self)
        self.btn4.clicked.connect(self.whichbtn2)
               
        hLayout2.addWidget(self.btn3)
        hLayout2.addWidget(self.btn4)
        

        self.pandasTv = QtWidgets.QTableView(self)
        hLayout.addWidget(self.pandasTv)
        vLayout.addLayout(hLayout)   
        
        self.pandasTv2 = QtWidgets.QTableView(self)
        hLayout2.addWidget(self.pandasTv2)
        vLayout.addLayout(hLayout2)
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        hLayout.addWidget(self.canvas)
        vLayout.addLayout(hLayout)
        
        self.fig2 = plt.Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        hLayout2.addWidget(self.canvas2)
        vLayout2.addLayout(hLayout2)
    
    def clickMethod(self):
        x= self.line.text()
        import statistics
        import requests
        import time
        import re
        from bs4 import BeautifulSoup as bs
        import pandas as pd
        def cleanText (readData):
            text =re.sub("[,""'\n\t]","",readData)
            return text
        def get_html(url):
            _html = ""
            resp = requests.get(url)
            if resp.status_code == 200:
                _html = resp.text
                return _html

        table = pd.read_csv("여행비용.csv", encoding='cp949',engine='python')
        val=round(float(table.values[7,1]))
        pd.options.display.float_format='{:.2f}'.format

        search = x
        naver_url = "https://search.shopping.naver.com/search/all.nhn?query="+ search
        naver_html=get_html(naver_url)
        naver_soup=bs(naver_html,'html.parser',from_encoding='utf-8')
        green_window = naver_soup.find_all('span', attrs={'num _price_reload'})
        prices=[]
        prices2=[]
        for ji in range(len(green_window)):
            prices.append(green_window[ji].text)

        for jj in range(len(prices)):
            prices2.append(int(cleanText(prices[jj])))

        product_price=statistics.median_high(prices2)

        remaning = product_price-val
        pie2 = ([product_price, val,remaning])
        pie_data2 = pd.DataFrame(pie2, columns=['얼마안남았어!'], index=["목표상품금액", "벌써 이만큼이나?","남은금액"])
        pd.options.display.float_format='{:.2f}'.format
        pie_data2.to_csv("목표상품.csv", sep=',', encoding='cp949')
    
    def whichbtn(self):
        df = pd.read_csv("목표상품.csv", encoding='cp949',engine='python', index_col=0)
        print(df)
        model = PandasModel(df)
        self.pandasTv.setModel(model)
        self.pandasTv.resizeColumnsToContents()
        self.pandasTv.update()
        
        import matplotlib.font_manager as fm
        from matplotlib import font_manager, rc
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        
        product_price = df.values[0,0]
        table = pd.read_csv("여행비용.csv", encoding='cp949',engine='python')
        val=round(float(table.values[7,1]))
        percentage = (val/product_price)*100
        resident_val = 100-percentage
        
        
        ax = self.fig.add_subplot(111)
        # Pie chart
        labels = ['남은금액은!','모을수 있어!']
        sizes = [resident_val, percentage]
        # only "explode" the 2nd slice (i.e. 'Hogs')
        explode = (0.1, 0)  
        ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=150)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')  
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.show()
        
        
        
    
    def clickMethod2(self):
        money = self.line2.text()
        table2 = pd.read_csv("여행비용.csv", encoding='cp949',engine='python')
        val2=round(float(table2.values[7,1]))
        pd.options.display.float_format='{:.2f}'.format
        remaning2 = float(money)-val2
        pie = ([money, val2,remaning2])
        
        
        pie_data = pd.DataFrame(pie, columns=['얼마안남았어!'], index=["목표금액", "벌써 이만큼이나?","남은금액"])
        pd.options.display.float_format='{:.2f}'.format
        pie_data.to_csv("목표금액.csv", sep=',', encoding='cp949')
    
    def whichbtn2(self):
        df = pd.read_csv("목표금액.csv", encoding='cp949',engine='python', index_col=0)
        print(df)
        model = PandasModel(df)
        self.pandasTv2.setModel(model)
        self.pandasTv2.resizeColumnsToContents()
        self.pandasTv2.update()
        
        import matplotlib.font_manager as fm
        from matplotlib import font_manager, rc
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        
        money = int(df.values[0,0])
        table = pd.read_csv("여행비용.csv", encoding='cp949',engine='python')
        val=round(float(table.values[7,1]))
        percentage2 = (val/money)*100
        resident_val2 = 100-percentage2
        
        ax2 = self.fig2.add_subplot(111)
        # Pie chart
        labels2 = ['남은금액은!','모을수 있어!']
        sizes2 = [resident_val2, percentage2]
        # only "explode" the 2nd slice (i.e. 'Hogs')
        explode2 = (0.1, 0)  
        ax2.pie(sizes2, explode=explode2, labels=labels2, autopct='%1.1f%%',
                shadow=True, startangle=150)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax2.axis('equal')
        self.fig2.tight_layout()
        self.canvas2.draw()
        self.canvas2.show()
        
        

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

