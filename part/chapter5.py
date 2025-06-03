import numpy as np
import cv2 as cv 
L = 256

def FrequencyFiltering(imgin,H):
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


def CreateMotionFilter(M,N):
    H = np.ones((M,N),np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    for u in range(M):
        for v in range(N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(phi) < 1.0e-6:
                RE = T
                IM = 0.0
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
    return H

def CreateMotion(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    H = CreateMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def CreateDeMotionFilter(M,N):
    H = np.ones((M,N),np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.00
    for u in range(M):
        for v in range(N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            temp = np.sin(phi)
            if abs(temp) < 1.0e-6:
                phi = phi_prev
            RE = phi/(T*np.sin(phi))*np.cos(phi)
            IM = phi/T
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def DeMotion(imgin):
    if len(imgin.shape) == 3:
        imgin = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    M,N = imgin.shape
    H = CreateDeMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout