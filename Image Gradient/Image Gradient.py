import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('Example.png',0)

dtype_out_cv_8u = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=5)

dtype_cv_64F= cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
sobel64f = np.absolute(dtype_cv_64F)
sobel_8u = np.uint8(sobel64f)

plt.subplot(1,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(dtype_out_cv_8u,cmap = 'gray')
plt.title('Image Gradient'), plt.xticks([]), plt.yticks([])
plt.savefig('Ex_ig.png')
plt.show()
