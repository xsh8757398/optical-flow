import cv2
import numpy as np
cap = cv2.VideoCapture("C:/Users/du/Desktop/testVideo.MP4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #上一个8位图像
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)  #第二张图像
    
    #0.5经典金字塔比例 下一层是前一层的二分之一
    #3 金字塔层数
    #15 值越高代表鲁棒性越高 快速运动检测越好 但是也更模糊
    #3 金字塔在每层的迭代次数
    #5 像素邻域的扩张 一般5/7
    #1.2 和上一个5对应
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    #计算二维量的大小和角度 x坐标数组 y坐标数组
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    #数组标准化  normType标准化类型 密集阵列都是它
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('Method_2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 9:
        break
    #elif k == ord('s'):
        #cv2.imwrite('opticalfb.png',frame2)
        #cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()
