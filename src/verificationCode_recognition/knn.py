#########################################  
# kNN: k Nearest Neighbors  
  
# Input:      inX: vector to compare to existing dataset (1xN)  
#             dataSet: size m data set of known vectors (NxM)  
#             labels: data set labels (1xM vector)  
#             k: number of neighbors to use for comparison   
              
# Output:     the most popular class label  
#########################################  
  
import numpy as np  
import operator  
import os  
import cv2

# classify using kNN  
def kNNClassify(input_set,data_sets,data_labels,k=10):
    # 获取训练集的行数,假设为m
    data_set_size = data_sets.shape[0]      
    
    # 创建一个行数为m的矩阵，每行数据与输入数据(测试数据)相同     
    new_input_set = np.tile(input_set,(data_set_size,1))    
    
    # 差矩阵(上面两矩阵相减)
    diff_matrix = new_input_set - data_sets         
    
    # 平方
    sq_diff_matrix = diff_matrix**2   
    
    # 距离: 先求平方和，再开方
    distance = (sq_diff_matrix.sum(axis=1))**0.5  
    
    # 将距离由小到大排序，并返回对应的序号集
    sort_distance_index = distance.argsort()
    
    pre_labels = {}
    for i in range(k):
        label = data_labels[sort_distance_index[i]]
        pre_labels[label] = pre_labels.get(label,0) + 1
    sorted_pre_labels = sorted(pre_labels.iteritems(),key=lambda x:x[1],reverse=True)
    
    return sorted_pre_labels[0][0]
  
# convert image to vector  
def  img2vector(img):
    img = cv2.imread(img,0)  #直接读取灰度图片
    rows,cols = img.shape
    imgVector = np.zeros([rows*cols],np.uint8)
    for i in range(rows):
        for j in range(cols):
            if img[i,j] > 200:
                imgVector[i * 32 + j] = 1
            else :
                imgVector[i * 32 + j] = 0 
  
    return imgVector  
  
# load dataSet  
def loadDataSet(pic_path='E:\\python_space\\segments\\train_pic'):  
    # 训练集
    training_labels = []
    dirList =  os.listdir(pic_path)  
    
    training_nums = 0
    #计算训练集样本数
    for dirName in dirList:  
        dirPath = os.path.join(pic_path, dirName)
        if os.path.isdir(dirPath):
            imgList = os.listdir(dirPath)
            training_nums += len(imgList)
    
    training_sets = np.zeros([training_nums,1024],np.float32) 
    
    i = 0
    for dirName in dirList:  
        dirPath = os.path.join(pic_path, dirName)
        if os.path.isdir(dirPath):
            imgList = os.listdir(dirPath)
            for imgName in imgList:
                imgPath = os.path.join(dirPath,imgName)
                training_sets[i,:] = img2vector(imgPath)
                training_labels.append(dirName)
                i += 1
                
    return  training_sets,training_labels  

def score(pred_labels,test_labels):
    pred = np.array(pred_labels)
    test = np.array(test_labels)
    res = (pred==test).astype(int)
    return res.sum()*1.0/res.shape[0]

# test hand writing class
if __name__ == '__main__':  
    print("获取训练集")
    traininig_sets , traininig_labels= loadDataSet('E:\\python_space\\segments\\train_pic')
    print("获取测试集")
    test_sets , test_labels = loadDataSet('E:\\python_space\\segments\\test_pic')
    pred_labels = []
    print("预测中...")
    
    i = 0
    for test in test_sets:
        pred_tag = kNNClassify(test,traininig_sets,traininig_labels,k=5)
        pred_labels.append(pred_tag)
        #print '预测：%s,实际：%s' %(pred_tag,test_labels[i])
        i += 1
        
    print ("准确率为:")
    print(score(pred_labels,test_labels))  
