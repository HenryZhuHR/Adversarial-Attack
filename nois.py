import cv2
import numpy as np

def add_noise(
    src:np.ndarray,
    sigma:float=0.0,
    sprate:float=0.0
):
    height,width,channel=src.shape


img:np.ndarray=cv2.imread('data/whu.jpg')
add_noise(img)