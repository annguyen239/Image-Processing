import cv2 as cv
import numpy as np
L = 256

# Xử lí ảnh morphology chủ yếu là xử lí ảnh nhị phân
# Ảnh nhị phân là ảnh chir có 2 màu là đen và trắng
def Erosion(imgin):
    w = cv.getStructuringElement(cv.MORPH_RECT, (45,45))
    imgout = cv.erode(imgin,w)
    return imgout

def Dilation(imgin):
    w = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    imgout = cv.dilate(imgin,w)
    return imgout 

def Boundary(imgin):
    w = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    temp = cv.erode(imgin,w)
    imgout = imgin - temp
    return imgout

def Contour(imgin):
    imgout = cv.cvtColor(imgin,cv.COLOR_GRAY2BGR)
    contours, _ = cv.findContours(imgin,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    n = len(contour)
    for i in range(n-1):
        x1 = contour[i,0,0]
        y1 = contour[i,0,1]
        x2 = contour[i+1,0,0]
        y2 = contour[i+1,0,1]
        cv.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
        
    x1 = contour[n-1,0,0]
    y1 = contour[n-1,0,1]
    x2 = contour[0,0,0]
    y2 = contour[0,0,1]
    cv.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    return imgout 

def ConvexHull(imgin):
    imgout=cv.cvtColor(imgin,cv.COLOR_GRAY2BGR)
    #b1: calculate contour
    contours,_=cv.findContours(imgin,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contour=contours[0]
    #b2: calculate convex hull
    hull=cv.convexHull(contour,returnPoints=False)
    n=len(hull)
    for i in range (0,n-1):
        vi_tri_1=hull[i,0]
        vi_tri_2=hull[i+1,0]
        x1=contour[vi_tri_1,0,0]
        y1=contour[vi_tri_1,0,1]
        x2=contour[vi_tri_2,0,0]
        y2=contour[vi_tri_2,0,1]
        cv.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    vi_tri_1=hull[n-1,0]
    vi_tri_2=hull[0,0]
    x1=contour[vi_tri_1,0,0]
    y1=contour[vi_tri_1,0,1]
    x2=contour[vi_tri_2,0,0]
    y2=contour[vi_tri_2,0,1]
    cv.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    return imgout

def Defectdetect(imgin):
    imgout=cv.cvtColor(imgin,cv.COLOR_GRAY2BGR)
    #b1: calculate contour
    contours,_=cv.findContours(imgin,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contour=contours[0]
    #b2: calculate convex hull
    hull=cv.convexHull(contour,returnPoints=False)
    n=len(hull)
    for i in range (0,n-1):
        vi_tri_1=hull[i,0]
        vi_tri_2=hull[i+1,0]
        x1=contour[vi_tri_1,0,0]
        y1=contour[vi_tri_1,0,1]
        x2=contour[vi_tri_2,0,0]
        y2=contour[vi_tri_2,0,1]
        cv.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    vi_tri_1=hull[n-1,0]
    vi_tri_2=hull[0,0]
    x1=contour[vi_tri_1,0,0]
    y1=contour[vi_tri_1,0,1]
    x2=contour[vi_tri_2,0,0]
    y2=contour[vi_tri_2,0,1]
    cv.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    
    #b3: defect detect
    defects = cv.convexityDefects(contour,hull)
    max_depth = np.max(defects[:,:,3])
    n = len(defects)
    for i in range(n):
        depth = defects[i,0,3]
        if depth > max_depth//2:
            pos = defects[i,0,2]
            x1 = contour[pos,0,0]
            y1 = contour[pos,0,1]
            cv.circle(imgout,(x1,y1),5, (0,255,0),-1)
    return imgout

def ConnectedComponents(imgin):
    nguong = 200
    _, temp = cv.threshold(imgin, nguong, L-1, cv.THRESH_BINARY)
    imgout = cv.medianBlur(temp,7)
    n, label = cv.connectedComponents(imgout, None)
    a = np.zeros(n,np.int32)    
    M,N = label.shape
    for x in range(M):
        for y in range(N):
            r = label[x,y]
            if r > 0:
                a[r] += 1
    s = 'Co %d thanh phan lien thong' % (n-1)
    cv.putText(imgout,s,(10,20),cv.FONT_HERSHEY_COMPLEX, 0.5,(255,255,255))
    for r in range(1,n):
        s = '%3d %4d' % (r,a[r])
        cv.putText(imgout,s,(10,(r+1)*20),cv.FONT_HERSHEY_COMPLEX, 0.5,(255,255,255))
    return imgout 

def RemoveSmallRice(imgin):
    w = cv.getStructuringElement(cv.MORPH_ELLIPSE, (81,81))
    temp = cv.morphologyEx(imgin, cv.MORPH_TOPHAT,w)
    nguong = 100
    _, temp = cv.threshold(temp,nguong,L-1,cv.THRESH_BINARY | cv.THRESH_OTSU)
    n,label = cv.connectedComponents(temp, None)
    a = np.zeros(n,np.int32)    
    M,N = label.shape
    for x in range(M):
        for y in range(N):
            r = label[x,y]
            if r > 0:
                a[r] += 1
    max_value = np.max(a)
    imgout = np.zeros((M,N),np.uint8)
    for x in range(M):
        for y in range(N):
            r = label[x,y]
            if r>0:
                if a[r] > 0.7*max_value:
                    imgout[x,y] = L-1
    return imgout