from PySide6.QtCore import (QCoreApplication, QMetaObject, Qt)
from PySide6.QtGui import ( QPainter,QPixmap,QPen,QPainterPath,)
from PySide6.QtWidgets import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

#网络结构
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net,self).__init__()
        #图片 1*28*28
        self.conv1 = nn.Conv2d(1,6,5) #24*24*20
        self.pool = nn.MaxPool2d(2,2) # 12*12*20
        self.conv2 = nn.Conv2d(6,16,3)# 10*10*40
        #5*5*40
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1,5*5*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#预测功能
class MNISTpredict():
    def __init__(self):
    #构建transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
        self.device = 'cuda'
        
        #创建模型
        self.net = CNN_Net().to(self.device)

        # 加载模型
        self.PATH = './res/mnist_cnn.pth'

        self.model = CNN_Net()

        self.model.load_state_dict(torch.load(self.PATH)) 

        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.model.load_state_dict(torch.load(self.PATH)) 
        self.model.eval()
        self.testgraph = cv2.imread('./res/tmp.png')

        # #裁剪大小
        self.testgraph = cv2.resize(self.testgraph,(28,28),interpolation=cv2.INTER_AREA)
        # 转换为灰度图
        self.testgraph = cv2.cvtColor(self.testgraph, cv2.COLOR_BGR2GRAY)
        #图像反色，对正确率影响巨大。因为mnist是黑底白字，而输入图片是白底黑字
        self.testgraph = cv2.bitwise_not(self.testgraph)
        #cv2.imwrite("./check.png",self.testgraph)
        # self.testgraph.show()

        self.testgraph = self.transform(self.testgraph)

        self.img_ = torch.unsqueeze(self.testgraph, 0)
        self.outputs = self.model(self.img_)

        #输出概率最大的类别 
        #图片笔画像素如果太低，不容易识别
        _, indices = torch.max(self.outputs,1)
        percentage = torch.nn.functional.softmax(self.outputs, dim=1)[0] * 100
        #perc = percentage[int(indices)].item()
        self.result = class_names[indices]
        #print('predicted:', result)


class MyScene(QGraphicsScene):#自定场景
    pen_color=Qt.black #预设笔的颜色
    pen_width=30 #预设笔的宽度
    def __init__(self):#初始函数
        super(MyScene, self).__init__() #实例化QGraphicsScene
        self.setSceneRect(0,0,400,400) #设置场景起始及大小，默认场景是中心为起始，不方便后面的代码

    def mousePressEvent(self, event):#重载鼠标事件
        if event.button() == Qt.LeftButton:#仅左键事件触发
            self.QGraphicsPath = QGraphicsPathItem() #实例QGraphicsPathItem
            self.path1 = QPainterPath()#实例路径函数
            self.path1.moveTo(event.scenePos()) #路径开始于
            pp=QPen() #实例QPen
            pp.setColor(self.pen_color) #设置颜色
            pp.setWidth(self.pen_width)#设置宽度
            self.QGraphicsPath.setPen(pp) #应用笔
            self.addItem(self.QGraphicsPath) #场景添加图元


    def mouseMoveEvent(self, event):#重载鼠标移动事件
        if event.buttons() & Qt.LeftButton: #仅左键时触发，event.button返回notbutton，需event.buttons()判断，这应是对象列表，用&判断
            if self.path1:#判断self.path1
                self.path1.lineTo(event.scenePos()) #移动并连接点
                self.QGraphicsPath.setPath(self.path1) #self.QGraphicsPath添加路径，如果写在上面的函数，是没线显示的，写在下面则在松键才出现线


    def mouseReleaseEvent(self, event):#重载鼠标松开事件
        if event.button() == Qt.LeftButton:#判断左键松开
            if self.path1:
                self.path1.closeSubpath() #结束路径


class Ui_Form(QWidget):
    def __init__(self):
        super(Ui_Form,self).__init__()
        self.formLayout = QFormLayout(self)
        self.formLayout.setObjectName(u"formLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.graphicsView = QGraphicsView(self)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setRenderHint(QPainter.Antialiasing) #设置反锯齿，注释掉曲线不平滑
        self.scene=MyScene()
        self.horizontalLayout.addWidget(self.graphicsView)
        #垂直展示板
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.verticalLayout.addItem(self.verticalSpacer)
        self.label1=QLabel()
        self.verticalLayout.addWidget(self.label1)
        #清除画布按钮
        self.pushButton_clear = QPushButton(self)
        self.pushButton_clear.setObjectName(u"pushButton_clear")
        self.verticalLayout.addWidget(self.pushButton_clear)
        #预测按钮
        self.pushButton_predict = QPushButton(self)
        self.pushButton_predict.setObjectName(u"pushButton_predict")
        self.verticalLayout.addWidget(self.pushButton_predict)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.verticalLayout.addItem(self.verticalSpacer_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.formLayout.setLayout(0, QFormLayout.SpanningRole, self.horizontalLayout)
        self.retranslateUi(self)
        QMetaObject.connectSlotsByName(self) #UI工具生成代码，注释看好像也没影响
        self.graphicsView.setScene(self.scene)

        self.pushButton_clear.clicked.connect(self.clean_all)
        self.pushButton_predict.clicked.connect(self.predict)



    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"机器学习课程设计-MNIST可视化识别        李欢欢、刘易行", None))
        self.label1.setText(QCoreApplication.translate("Form", u"请画出阿拉伯数字\n尽量填满屏幕", None))
        self.pushButton_clear.setText(QCoreApplication.translate("Form", u"清除内容", None))
        self.pushButton_predict.setText(QCoreApplication.translate("Form", u"预测数字", None))
    # retranslateUi

    def clean_all(self):#清除图元
        self.scene.clear()

    def predict(self):
        #先在本地保存一个临时的图像文件
        self.save()
        #然后将文件输入到网络中
        #再将网络预测结果输入至
        self.MNISTpredict = MNISTpredict()
        QMessageBox.information(
                                self,
                                '预测成功',
                                '该数字为'+str(self.MNISTpredict.result))


    def save(self):
        #保存graphicsView的图片
        # rect = self.graphicsView.scene().sceneRect()
        # pixmap = QImage(rect.height(),rect.width(),QImage.Format_ARGB32_Premultiplied)
        # painter = QPainter(pixmap)
        # rectf = QRectF(0,0,pixmap.rect().height(),pixmap.rect().width())
        # self.graphicsView.scene().render(painter,rectf,rect)
        # pixmap.save('./file.png')
        #必须加，不然报错
        # del painter
        # del pixmap
        
        view = self.graphicsView
        pixmap = QPixmap(view.viewport().size())
        view.viewport().render(pixmap)
        pixmap.save("./res/tmp.png")
        


if __name__=='__main__':
    app=QApplication([])
    MyWin=Ui_Form()
    MyWin.show()
    app.exec()