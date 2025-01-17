import cv2
import numpy as np
from IPython import display
from matplotlib import pyplot as plt

def imshow(img, format=".jpg", **kwargs):
    """ndarray 配列をインラインで Notebook 上に表示する。"""
    img = cv2.imencode(format, img)[1]
    img = display.Image(img, **kwargs)
    display.display(img)

file_name='images/20250117134627.jpg'
img=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)

h,w=img.shape[:2]
#二値化処理
thresh, img_thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
# cv2.imwrite('img_thresh.jpg',img_thresh)
imshow(img_thresh)
#輪郭抽出
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))