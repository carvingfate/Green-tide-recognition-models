# 准备数据集
import os
import cv2 as cv
import numpy as np
from osgeo import gdal
path1 = '/content/all/image' #图片文件夹
path2 = '/content/all/label' #标签文件夹
path_list1 = []
path_list2 = []
for file in os.listdir(path1):
        temp_dir = os.path.join(path1, file)  # /content/GEI_CASIA_B/gei/001
        path_list1.append(temp_dir)

for file in os.listdir(path2):
        temp_dir = os.path.join(path2, file)  # /content/GEI_CASIA_B/gei/001
        path_list2.append(temp_dir)
print(len(path_list1))
print(len(path_list2))
file_handle=open('1.txt',mode='w')
my_set = set()
for i in range(len(path_list1)):
        temp1 = gdal.Open(path_list1[i])
        width = temp1.RasterXSize
        height = temp1.RasterYSize
        temp1 = temp1.ReadAsArray(0, 0, width, height)
        #print(temp1.shape)
        temp2 = cv.imread(path_list2[i],-1)
        for j in range(temp1.shape[0]):
                for k in range(temp1.shape[1]):
                        if temp2[j][k][0] > 122:
                            temp2[j][k][0] = 255
                        else:
                            temp2[j][k][0] = 0
                        file_handle.write(str(temp1[j][k][0])+','+str(temp1[j][k][1])+','+str(temp1[j][k][2])+','+str(10*temp1[j][k][3])+','+str(temp2[j][k][0])+'\n')
                        my_set.add(temp1[j][k][0]+temp1[j][k][1]+temp1[j][k][2]+temp1[j][k][3])


# SVM
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import model_selection
import pickle
print("SVM")
#  定义字典，便于来解析样本数据集txt
def Iris_label(s):
    it={b'0':0, b'255':1}
    return it[s]

path=r"1.txt"
SavePath = r"SVM_model.pickle"


#  1.读取数据集
data=np.loadtxt(path, dtype=float, delimiter=',', converters={4:Iris_label} )
#  converters={7:Iris_label}中“7”指的是第8列：将第8列的str转化为label(number)

#  2.划分数据与标签
x,y=np.split(data,indices_or_sections=(4,),axis=1) #x为数据，y为标签
x=x[:,0:4] #选取前7个波段作为特征
train_data,test_data,train_label,test_label = model_selection.train_test_split(x,y, random_state=1, train_size=0.9,test_size=0.1)
new_train_label = []
new_train_data = []
for i in range(len(train_label)):
  #print(test_label[i,0])
  if train_label[i,0] == 1:
    #print("oui")
    new_train_label.append(train_label[i])
    new_train_data.append(train_data[i])
  elif i%10 == 9:
    new_train_label.append(train_label[i])
    new_train_data.append(train_data[i])

#3.训练svm分类器
#kernel='rbf'时，为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
from sklearn import svm
classifier=svm.SVC(C=1,kernel='rbf',gamma=2,decision_function_shape='ovo')
classifier.fit(new_train_data,new_train_label) #ravel函数拉伸到一维

#  4.计算随机森林的准确率
print("训练集：",classifier.score(new_train_data,new_train_label))
print("测试集：",classifier.score(test_data,test_label))

#  5.保存模型
#以二进制的方式打开文件：
file = open(SavePath, "wb")
#将模型写入文件：
pickle.dump(classifier, file)
#最后关闭文件：
file.close()

#计算各项指标
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
f1 = sklearn.metrics.f1_score(test_label, pred_label, labels=None, pos_label=1)
print("f1:",f1)
cm = confusion_matrix(y_true=test_label, y_pred=pred_label)
print("Confusion Matrix: ")
print(cm)

# 画出混淆矩阵
# ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_label)
#disp.plot()
#plt.show()
recall = sklearn.metrics.recall_score(test_label, pred_label)
print("recall:",recall)
intersection = np.diag(cm)  
union = np.sum(cm, axis = 1) + np.sum(cm, axis = 0) - np.diag(cm)  
IoU = intersection / union
print("iou:",IoU)
mIoU = np.nanmean(IoU)  
print("miou:",mIoU)
freq = np.sum(cm, axis=1) / np.sum(cm)  
iu = np.diag(cm) / (
np.sum(cm, axis = 1) + np.sum(cm, axis = 0) - np.diag(cm))
FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
print("fwiou:",FWIoU)

# 预测结果写tiff
import numpy as np
#import gdal
import pickle
import os
from osgeo import gdal
#定义图像打开方式
def image_open(image):
    data1 = gdal.Open(image)
    if data1 == None:
        print("数据无法打开")
    else:
        print("oui")
    return data1

#定义模型打开方式
def model_open(model):
    data1 = open(model, "rb")
    data2 = pickle.load(data1)
    data1.close()
    return data2

#定义图像保存
def writeTiff(im_data, im_geotrans, im_proj, path):
    #im_bands, im_height, im_width = im_data.shape
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = None
    if len(im_data.shape) == 3 or len(im_data.shape) == 2:
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

#定义相关参数
Image_FilePath = r"/content/test_input"
Model_FiLePath = r"/content/SVM_model.pickle"
Save_FilePath = r"/content/test_output"
model1 = model_open(Model_FiLePath)
for file in os.listdir(Image_FilePath):
        #print(file)
        src_dir = os.path.join(Image_FilePath, file)
        image1 = image_open(src_dir)
        print(image1)
        width = image1.RasterXSize
        height = image1.RasterYSize
        Projection = image1.GetProjection()
        Transform = image1.GetGeoTransform()
        image2 = image1.ReadAsArray()
        #在与测试前要调整一下数据的格式
        data = np.zeros((image2.shape[0], image2.shape[1] * image2.shape[2]))
        for i in range(image2.shape[0]):
            data[i] = image2[i].flatten()
        data = data.swapaxes(0, 1)

        #对调整好格式的数据进行预测
        print(data.shape)
        pred = model1.predict(data)

        #同样地，我们对预测好的数据调整为我们图像的格式
        pred = pred.reshape(image2.shape[1], image2.shape[2]) * 255
        pred = pred.astype(np.uint8)
        print("pred finish")
        #将结果写到tif图像里
        writeTiff(pred, Transform, Projection, os.path.join(Save_FilePath, file))
        print(0)
 