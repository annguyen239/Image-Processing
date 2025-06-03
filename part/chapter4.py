import numpy as np
import cv2 as cv 
L = 256

def Spectrum(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    fp = np.zeros((P,Q),np.float32)
    fp[:M,:N] = 1.0*imgin/(L-1)
    
    for i in range(M):
        for j in range(N):
            if (i+j)%2 == 1:
                fp[i,j] = -fp[i,j]
    
    F = cv.dft(fp,flags=cv.DFT_COMPLEX_OUTPUT)  
    
    FR = F[:,:,0]
    FI = F[:,:,1]
    Spectrum = np.sqrt(FR**2 + FI**2)
    Spectrum = np.clip(Spectrum,0,L-1)
    imgout = Spectrum.astype(np.uint8)
    return imgout
def CreateNotchFilter(P,Q):
    H = np.ones((P,Q,2),np.float32)
    H[:,:,1] = 0.0
    u1, v1 = 45,58
    u2,v2 = 86,58
    u3,v3 = 40,119
    u4,v4 = 82,119
    
    u5,v5 = P-45,Q-58
    u6,v6 = P-86,Q-58
    u7,v7 = P-40,Q-119
    u8,v8 = P-82,Q-119
    D0 = 15
    for u in range(P):
        for v in range(Q):
            # u1,v1
            D = np.sqrt(1.0*u-u1)**2 + (1.0*v-v1)**2
            if D < D0:
                H[u,v,0] = 0.0
            
            # u2,v2
            D = np.sqrt(1.0*u-u2)**2 + (1.0*v-v2)**2
            if D < D0:
                H[u,v,0] = 0.0
            
            # u3,v3
            D = np.sqrt(1.0*u-u3)**2 + (1.0*v-v3)**2
            if D < D0:
                H[u,v,0] = 0.0
            
            # u4,v4
            D = np.sqrt(1.0*u-u4)**2 + (1.0*v-v4)**2
            if D < D0:
                H[u,v,0] = 0.0
            
            # u5,v5
            D = np.sqrt(1.0*u-u5)**2 + (1.0*v-v5)**2
            if D < D0:
                H[u,v,0] = 0.0
            
            # u6,v6
            D = np.sqrt(1.0*u-u6)**2 + (1.0*v-v6)**2
            if D < D0:
                H[u,v,0] = 0.0
            
            # u7,v7
            D = np.sqrt(1.0*u-u7)**2 + (1.0*v-v7)**2
            if D < D0:
                H[u,v,0] = 0.0
            
            # u8,v8
            D = np.sqrt(1.0*u-u8)**2 + (1.0*v-v8)**2
            if D < D0:
                H[u,v,0] = 0.0
    return H

def DrawNotchFilter(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    H = CreateNotchFilter(P,Q)
    HR = H[:,:,0]*(L-1)
    imgout = HR.astype(np.uint8)
    return imgout 

def CreateNotchPeriodFilter(P,Q):
    H = np.ones((P,Q,2),np.float32)
    H[:,:,1 ] = 0.0
    D0 = 10
    for u in range(P):
        for v in range(Q):
            if u not in range(P//2-15,P//2+15+1):
                if abs(v-Q//2) <= D0:
                    H[u,v,0] = 0.0
    return H
                    
def DrawNotchPeriodFilter(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    H = CreateNotchPeriodFilter(P,Q)
    HR = (H[:,:,0])*(L-1)
    imgout = HR.astype(np.uint8)
    return imgout 

def RemoveMoire(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    fp = np.zeros((P,Q),np.float32)
    fp[:M,:N] = 1.0*imgin/(L-1) # bỏ L-1 sẽ cho ra ảnh xe
    
    for i in range(M):
        for j in range(N):
            if (i+j)%2 == 1:
                fp[i,j] = -fp[i,j]
    F = cv.dft(fp,flags=cv.DFT_COMPLEX_OUTPUT)
    
    H = CreateNotchFilter(P,Q)
    G = cv.mulSpectrums(F,H,flags=cv.DFT_ROWS)
    g = cv.idft(G,flags=cv.DFT_SCALE)
    
    gR = g[:M,:N,0]
    for i in range(M):
        for j in range(N):
            if (i+j)%2 == 1:
                gR[i,j] = -gR[i,j]
    gR = np.clip(gR,0,L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def RemovePeriodNoise(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    P = cv.getOptimalDFTSize(M)
    Q = cv.getOptimalDFTSize(N)
    fp = np.zeros((P,Q),np.float32)
    fp[:M,:N] = 1.0*imgin # bỏ L-1 sẽ cho ra ảnh xe
    
    for i in range(M):
        for j in range(N):
            if (i+j)%2 == 1:
                fp[i,j] = -fp[i,j]
    F = cv.dft(fp,flags=cv.DFT_COMPLEX_OUTPUT)
    
    H = CreateNotchPeriodFilter(P,Q)
    G = cv.mulSpectrums(F,H,flags=cv.DFT_ROWS)
    g = cv.idft(G,flags=cv.DFT_SCALE)
    
    gR = g[:M,:N,0]
    for i in range(M):
        for j in range(N):
            if (i+j)%2 == 1:
                gR[i,j] = -gR[i,j]
    gR = np.clip(gR,0,L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def FrequencyFiltering(imgin,H):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    f = imgin.astype(np.float64)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)
    G = F * H
    G = np.fft.ifftshift(G)
    g = np.fft.ifft2(G)
    gR = g.real.copy()
    gR = np.clip(gR,0,L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def Spec(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    f = imgin.astype(np.float32)/(L-1)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)
    S = np.sqrt(F.real**2 + F.imag**2)
    S = np.clip(S,0,L-1)

    imgout = S.astype(np.uint8)
    return imgout
    
def CreateNotchFilterFreq(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag= 0.0
    u1, v1 = 44,55
    u2,v2 = 85,55
    u3,v3 = 40,111
    u4,v4 = 84,111
    
    u5,v5 = M-44,N-55
    u6,v6 = M-85,N-55
    u7,v7 = M-40,N-111
    u8,v8 = M-84,N-111
    D0 = 15
    for u in range(M):
        for v in range(N):
            # u1,v1
            D = np.sqrt(1.0*u-u1)**2 + (1.0*v-v1)**2
            if D < D0:
                H.real[u,v] = 0.0
            
            # u2,v2
            D = np.sqrt(1.0*u-u2)**2 + (1.0*v-v2)**2
            if D < D0:
                H.real[u,v] = 0.0
            
            # u3,v3
            D = np.sqrt(1.0*u-u3)**2 + (1.0*v-v3)**2
            if D < D0:
                H.real[u,v] = 0.0
            
            # u4,v4
            D = np.sqrt(1.0*u-u4)**2 + (1.0*v-v4)**2
            if D < D0:
                H.real[u,v] = 0.0
            
            # u5,v5
            D = np.sqrt(1.0*u-u5)**2 + (1.0*v-v5)**2
            if D < D0:
                H.real[u,v] = 0.0
            
            # u6,v6
            D = np.sqrt(1.0*u-u6)**2 + (1.0*v-v6)**2
            if D < D0:
                H.real[u,v] = 0.0
            
            # u7,v7
            D = np.sqrt(1.0*u-u7)**2 + (1.0*v-v7)**2
            if D < D0:
                H.real[u,v] = 0.0
            
            # u8,v8
            D = np.sqrt(1.0*u-u8)**2 + (1.0*v-v8)**2
            if D < D0:
                H.real[u,v] = 0.0
    return H

def RemoveMoireFreq(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    H = CreateNotchFilterFreq(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def CreateNotchInterferenceFilter(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag= 0.0
    D0 = 7
    D1 = 7
    for u in range(M):
        for v in range(N):
            if u not in range(M//2-D1,M//2+D1+1):
                if v in range(N//2-D0,N//2+D0+1):
                    H.real[u,v] = 0.0
    return H

def RemoveInterferenceFreq(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    H = CreateNotchInterferenceFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout    

