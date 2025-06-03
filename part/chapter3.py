import cv2 as cv
import numpy as np

L = 256
def Negative(imgin):
    # If the image is colored, convert it to grayscale first
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + np.uint8(255)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            s = 255 - r
            imgout[x, y] = np.uint8(s)
    return imgout

def NegativeColor(imgin):
    # Check if image is grayscale, then convert to color
    if len(imgin.shape) == 2:
        imgin = cv.cvtColor(imgin, cv.COLOR_GRAY2BGR)
    
    M, N, C = imgin.shape    
    imgout = np.zeros((M, N, C), np.uint8)
    
    # Process each channel (BGR)
    for x in range(M):
        for y in range(N):
            b = imgin[x, y, 0]
            g = imgin[x, y, 1]
            r = imgin[x, y, 2]
            
            imgout[x, y, 0] = np.uint8(255 - b)
            imgout[x, y, 1] = np.uint8(255 - g)
            imgout[x, y, 2] = np.uint8(255 - r)
    return imgout

def Log(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape    
    imgout = np.zeros((M,N), np.uint8)
    c = (L-1.0)/np.log(1.0*L)
    # Quét ảnh
    for x in range(M):
        for y in range(N):
            r = imgin[x,y]
            if r == 0:
                r = 1
                
            s = c*np.log(1.0 + r)
            imgout[x,y] = np.uint8(s)
    return imgout

def Power(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape    
    imgout = np.zeros((M,N), np.uint8)
    gamma = 5.0
    c = np.power(L-1.0,1.0-gamma)
    for x in range(M):
        for y in range(N):
            r = imgin[x,y]
            if r == 0:
                r = 1
            s = c*np.power(1.0*r,gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

def PiecewiseLine(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M, N = imgin.shape    
    imgout = np.zeros((M, N), np.uint8)
    rmin, rmax, _, _ = cv.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1  # s2 = 255

    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            if r < r1:
                # Avoid division by zero if r1 is zero
                if r1 != 0:
                    s = 1.0 * s1 / r1 * r
                else:
                    s = s1
            elif r < r2:
                if r2 != r1:
                    s = 1.0 * (s2 - s1) / (r2 - r1) * (r - r1) + s1
                else:
                    s = s1
            else:
                # Check for division by zero in the else branch
                if (L - 1 - r2) != 0:
                    s = 1.0 * (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2
                else:
                    s = s2
            imgout[x, y] = np.uint8(s)
    return imgout


def Histogram(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M,L,3), np.uint8) + np.uint8(255)
    h = np.zeros(L,np.int32)
    for x in range(M):
        for y in range(N):
            r = imgin[x,y]
            h[r] = h[r] + 1
    p = 1.0*h/(M*N)
    scale = 3000
    for r in range(L):
        cv.line(imgout, (r,M-1), (r,M-1-np.int32(scale*p[r])),(255, 0, 0))
    return imgout

def HistEqual(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) 
    h = np.zeros(L,np.int32)
    for x in range(M):
        for y in range(N):
            r = imgin[x,y]
            h[r] = h[r] + 1
    p = 1.0*h/(M*N)
    
    s = np.zeros(L,np.float64)
    for k in range(L):
        for j in range(k+1):
            s[k] = s[k] + p[j]
    for x in range(M):
        for y in range(N):
            r = imgin[x,y]
            imgout[x,y] = np.uint8((L-1)*s[r])
    return imgout

def LocalHist(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    imgout = np.zeros((M,N),np.uint8)
    m = 3
    n = 3
    a = m//2
    b = n//2
    for x in range(a, M-a):
        for y in range(b, M-b):
            w = imgin[x-a:x+a+1,y-b:y+b+1]
            w = cv. equalizeHist(w)
            imgout[x,y] = w[a,b]
    return imgout

def HistStat(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    imgout = np.zeros((M,N),np.uint8)
    mean, stddev = cv.meanStdDev(imgin)
    mG = mean[0,0]
    sigmaG = stddev[0,0]
    
    print("Mean: ",mean)
    print("Stddev: ", stddev)
    
    m = 3
    n = 3
    a = m//2
    b = n//2
    
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M-a):
        for y in range(b, M-b):
            w = imgin[x-a:x+a+1,y-b:y+b+1]
            mean, stddev = cv.meanStdDev(w)
            msxy = mean[0,0]
            sigmasxy = stddev[0,0]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy < k3*sigmaG):
                imgout[x,y] = np.uint8(C*imgin[x,y])
            else:
                imgout[x,y] = imgin[x,y]
    return imgout

def BoxFilter(imgin):
    m = 21
    n = 21
    w = np.zeros((m,n),np.float32) + np.float32(1.0/(m*n))
    imgout = cv.filter2D(imgin, cv.CV_8UC1, w)
    return imgout

def smoothGauss(imgin):
    m = 43
    n = 43
    sigma = 7.0
    a = m // 2
    b = n // 2

    w = np.zeros((m, n), np.float32)
    for s in range(-a,a+1):
        for t in range(-b,b+1):
            w[s+a,t+b] = np.exp(-(s**2 + t**2) /(2.0*sigma**2))

    K = np.sum(w)
    w = w / K
    imgout = cv.filter2D(imgin, cv.CV_8UC1, w)
    return imgout

def Sharp(imgin):
    w = np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]], np.float32)
    Laplacian = cv.filter2D(imgin, cv.CV_32FC1, w)
    imgout = imgin - Laplacian
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Gradient(imgin):
    Sobel_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float32)
    Sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,-1]],np.float32)
    gx = cv.filter2D(imgin, cv.CV_32FC1, Sobel_x)
    gy = cv.filter2D(imgin, cv.CV_32FC1, Sobel_y)
    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout