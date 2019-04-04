import cv2      
import numpy as np 
import matplotlib.pyplot as plt  
import os
import math

## 计算在X坐标投影的非0像素个数 ##
def calXProjection(image,isSmooth):
    rows,cols = image.shape
    hist = np.zeros([cols],np.uint16)
    
    for i in range(cols):
        for j in range(rows):
            if image[j,i] > 200:
                hist[i] += 1
    
    if isSmooth :
        for i in range(cols):
            if hist[i] > 5:
                hist[i] = (hist[i]+hist[(i+1 + cols)%cols]+hist[(i+2+cols)%cols]+hist[(i-1 + cols)%cols]+hist[(i-2+cols)%cols])/5
            else :
                hist[i] = 0
        for i in range(cols):
            if hist[(i-2 + cols)%cols] == 0 and hist[(i+2 + cols)%cols] == 0:
                hist[i] = 0
                hist[(i-1 + cols)%cols] = 0
                hist[(i+1 + cols)%cols] = 0
    
    ### draw hist image for test  ###
#     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
#     barH = 100
#     histImg = np.zeros([barH+rows,cols], np.uint8)
#     histImg[barH:barH+rows,:] = image
#     hpt = int(0.9* barH)   
#     for h in range(cols):    
#         intensity = int(hist[h]*hpt/maxVal)    
#         cv2.line(histImg,(h,barH), (h,barH-intensity), 255)
#     histImgBig = cv2.resize(histImg,(2*cols, 2*(rows+barH)), interpolation = cv2.INTER_CUBIC)
#     cv2.imshow("Img_hist", histImgBig) 
#     cv2.waitKey(0)    
#     cv2.destroyAllWindows()
    return hist
    
    ## 计算在Y坐标投影的非0像素个数 ##
def calYProjection(image,isSmooth):
    rows,cols = image.shape
    hist = np.zeros([rows],np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if image[i,j] > 200:
                hist[i] += 1
    
    if isSmooth :
        for i in range(rows):
            if hist[i] > 2:
                hist[i] = (hist[i]+hist[(i+1 + rows)%rows]+hist[(i+2+rows)%rows]+hist[(i-1 + rows)%rows]+hist[(i-2+rows)%rows])/5
            else :
                hist[i] = 0
        for i in range(rows):
            if hist[(i-2 + rows)%rows] == 0 and hist[(i+2 + rows)%rows] == 0:
                hist[i] = 0
                hist[(i-1 + rows)%rows] = 0
                hist[(i+1 + rows)%rows] = 0

    return hist
    
    
def getROI(img):
    h, w = img.shape
    ##图像二值化##
    #Gaussian filtering
    img_seg = img[8:h,40:w-40]
    height, width = img_seg.shape
    
    img_blur = cv2.GaussianBlur(img_seg,(5,5),0)

    #自适应高斯二值化
    img_thre = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,11,3)
    #腐蚀
    kernel = np.ones((2,1),np.uint8)
    erosion = cv2.erode(img_thre, kernel, iterations = 3)
    
     #垂直方向膨胀
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    
    #计算像素x轴投影
    hist = calXProjection(dilation,True)
    
    #根据投影判断开始和结束的位置
    start_pos = 0
    end_pos = width - 1
    for i in range(width - 4):
        if hist[i]+hist[i+1]+hist[i+2]+hist[i+3]+hist[i+4] > 25:
            start_pos = i+1
            break
    for i in range(width - 4):
        if hist[width - i -1]+hist[width - i -2]+hist[width - i -3]+hist[width - i -4]+hist[width - i -5] > 5:
            end_pos = width - i
            break
    
    img_ROI = dilation[1:height,start_pos:end_pos]
    return img_ROI
 
    
# 计算字符边界，boundSize字符个数
def charXBoundary(img,boundSize = 5):
    height,width = img.shape
    partWidth = round((width *1.0) / (boundSize *1.0))      # 单个字符初始宽度
    bounds = np.zeros([boundSize+1,height],np.uint16)       # 字符边界
    energy_bounds = np.zeros([boundSize],np.uint16)         # 计算能量的边界，用于计算最终的字符边界
    accu_energys = np.zeros([height, width], np.uint32)     # 能量累加矩阵
    accu_routes = np.zeros([height, width], np.int8)        # 计算能量累加的路线
    
    #初始化字符的左右边界，（假设字符等宽，以中间字符左右对称计算）
    size = int(boundSize/2)
    if boundSize %2 == 0:
        size -= 1
    for i in range(size+1):
        for j in range(height):
            bounds[i][j] = i*partWidth
            bounds[boundSize-i][j] = width - i*partWidth  
    if boundSize%2 == 0:
        for j in range(height):
            bounds[size+1][j] = round(width/2)

    # 累加能量左右边界初始化（左右边界初始化为字符的中心） 
    for i in range(boundSize):
        energy_bounds[i] = round((bounds[i][0] + bounds[i+1][0])/2)
    
    #计算累加能量矩阵
    for i in range(energy_bounds[0],energy_bounds[boundSize-1]):
        accu_energys[0][i] = img[0][i]
        
    for i in range(1,height):
        for j in range(boundSize - 1):
            for k in range(energy_bounds[j],energy_bounds[j+1]):
                if k == energy_bounds[j]:
                    if accu_energys[i-1][k] <= accu_energys[i-1][k+1]:
                        accu_energys[i][k] = img[i][k] + accu_energys[i-1][k]
                    else:
                        accu_energys[i][k] = img[i][k] + accu_energys[i-1][k+1]
                        accu_routes[i][k] = 1
                elif k == (energy_bounds[j+1] - 1):
                    if accu_energys[i-1][k] <= accu_energys[i-1][k-1]:
                        accu_energys[i][k] = img[i][k]+accu_energys[i-1][k]
                    else:
                        accu_energys[i][k] = img[i][k] + accu_energys[i-1][k-1]
                        accu_routes[i][k] = -1
                else :
                    temp  = getMin(accu_energys[i-1][k-1], accu_energys[i-1][k], accu_energys[i-1][k+1])
                    if temp == 0:
                        accu_energys[i][k] = img[i][k]+accu_energys[i-1][k]
                    elif temp == -1:
                        accu_energys[i][k] = img[i][k] + accu_energys[i-1][k-1]
                        accu_routes[i][k] = -1
                    else:
                        accu_energys[i][k] = img[i][k] + accu_energys[i-1][k+1]
                        accu_routes[i][k] = 1
    
    #寻找能量最小的字符边界
    for j in range(boundSize - 1):
        minInt = height*255
        minLoc = energy_bounds[j]
        for k in range(energy_bounds[j],energy_bounds[j+1]):
            if accu_energys[height-1][k] < minInt:
                minInt = accu_energys[height-1][k]
                minLoc = k
            if accu_energys[height-1][k] == minInt:
                if abs(minLoc - bounds[j+1][0]) > abs(k - bounds[j+1][0]):
                    minLoc = k
                    minInt = accu_energys[height-1][k]
        bounds[j+1][height-1] = minLoc
        
    for j in range(boundSize - 1):
        for i in range(1,height):
            bounds[j+1][height-i - 1] = bounds[j+1][height-i] + accu_routes[height-i][bounds[j+1][height-i]]
           
    return  bounds    
                
                
def getMin(value1,value2,value3):
    if value2 <= value1 and value2 <= value3:
        return 0
    elif value1 <= value2 and value1 <= value3:
        return -1
    else:
        return 1
    
    
def charCropping(img):
    height,width = img.shape
     #计算像素x轴投影
    hist_x = calXProjection(img,False)
    hist_y = calYProjection(img,True)
    
    start_x = 0
    end_x = width - 1
    start_y = 0
    end_y = height - 1
    
    for i in range(width - 2):
        if hist_x[width - i - 1]+hist_x[width -i-2]+hist_x[i-3] > 5:
            end_x = width -i
            break
            
    for i in range(width - 2):
        if hist_x[width-i-1]+hist_x[width - i -2]+hist_x[width - i -3] > 5:
            start_x = width - i -3
       
    for i in range(height - 3):
        if hist_y[i]+hist_y[i+1]+hist_y[i+2]+hist_y[i+3] > 10:
            start_y = i
            break
    img_res = img[start_y:height,start_x:end_x]
    return img_res
    
    
## 字符切割的主函数 
def charSegment(img,boundSize):
    height,width = img.shape
    #能量累加法确定图像各个字符之间的边界
    bounds = charXBoundary(img,boundSize)
    vector_sets = np.zeros([boundSize,1024],np.float32)

    #字符中间边界x轴上的最大最小值
    max_min_Bounds = np.zeros([boundSize-1,2],np.uint16)
    for j in range(boundSize - 1):
        max_min_Bounds[j][1] = width
        for i in range(height):
            if bounds[j+1][i] > max_min_Bounds[j][0]:
                max_min_Bounds[j][0] = bounds[j+1][i]
            if bounds[j+1][i] < max_min_Bounds[j][1]:
                max_min_Bounds[j][1] = bounds[j+1][i]
             
    for j in range(boundSize):
        w = 0
        start_loc = 0
        if j  == 0:
            w = max_min_Bounds[j][0]
        elif j == boundSize - 1:
            w = abs(width - max_min_Bounds[j-1][1])
            start_loc =  max_min_Bounds[j-1][1]         
        else:
            w = abs(max_min_Bounds[j][0] - max_min_Bounds[j-1][1])
            start_loc =  max_min_Bounds[j-1][1]  
            
        char_img = np.zeros([height,w],np.uint8)
        for m in range(height):
            #print("j:%d, m:%d, start_loc:%d" % (j,m,start_loc))
            for n in range(w):
                ori_loc = start_loc + n
                if ori_loc < bounds[j+1][m] and ori_loc >= bounds[j][m]:
                    char_img[m][n] =img[m][ori_loc]
        if w<= 4:
            continue
        char_img_crop = charCropping(char_img)
        char_img_resize = cv2.resize(char_img_crop,(22, 32), interpolation = cv2.INTER_CUBIC)
        res_img = np.zeros([32,32], np.uint8)
        res_img[0:32,5:27] = char_img_resize
        
        for x in range(32):
            for y in range(32):
                if res_img[x,y] > 200:
                    vector_sets[j, x * 32 + y] = 1.0
                else:
                    vector_sets[j, x * 32 + y] = 0.0
  
    return vector_sets  