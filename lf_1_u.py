import numpy as np
import cv2
from time import clock
 
lk_params = dict( winSize  = (15, 15),    #搜索窗口大小 每一个金字塔
                  maxLevel = 2,           #最大的金字塔层数 3层
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    
 
feature_params = dict( maxCorners = 50,    #最多的数量 如果大于 就返回最明显的一些
                       qualityLevel = 0.3,  
                       minDistance = 7,     #角点之间最短欧氏距离
                       blockSize = 7 )
 
class App:
    def __init__(self, video_src):#构造方法，初始化一些参数和视频路径
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
 
    def run(self):#光流运行方法
        while True:
            ret, frame = self.cam.read()#读取视频帧
            if ret == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转化为灰度虚图像
                vis = frame.copy()
    
                if len(self.tracks) > 0:#检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    
                    #前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    #创建一个应用跟踪视频中的点
                    '''
                    p1输出跟踪特征点向量  st特征点是否找到 找到为1  err输出错误向量
                    img0前一帧8bit图像  img1当前帧8bit图像
                    '''
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                    #当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    #需要传入前一帧，前一帧点集，下一帧，找到点集，否者为0.
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                    #得到角点回溯与前一帧实际角点的位置变化关系
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    
                    good = d < 1#判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    new_tracks = []
                    
                    #将跟踪正确的点列入成功跟踪点
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    
                    #以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
    
                if self.frame_idx % self.detect_interval == 0:#每5帧检测一次特征点
                    mask = np.zeros_like(frame_gray)#初始化和视频大小相同的图像
                    mask[:] = 255#将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:#跟踪的角点画圆
                        cv2.circle(mask, (x, y), 5, 0, -1)
                        
                    #获取需要跟踪的点。通过第一帧取一些Shi-Tomasi 角点，然后用Lucas-Kanade迭代跟踪这些点
                    '''
                    初始特征点，即角点
                    输入的单通道图像
                    最大角点数
                    最小可接受的角点质量
                    角点间的最小欧几里得距离（两个角点不能太近）
                    '''
                    #像素级别角点检测
                    #图像 角点
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])#将检测到的角点放在待跟踪序列中
    
    
                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('Method_1', vis)
 
            ch = 0xFF & cv2.waitKey(1)
            if ch == 9:
                break
 
def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = "C:/Users/du/Desktop/testVideo.mp4"
 
    print (__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()             
 
if __name__ == '__main__':
    main()
