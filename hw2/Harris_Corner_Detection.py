import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage

def gaussian_smooth(size, sigma=1):
    # 建立一個一維的座標陣列，並將中心點設為 0
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    
    # 計算 2D 高斯分佈
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    # 正規化確保總和為 1
    img = kernel / np.sum(kernel)
    return img



def sobel_edge_detection(im, *args):
    # 定義 Sobel filters
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # 透過卷積計算 x 與 y 方向的梯度
    Ix = ndimage.convolve(im, Kx)
    Iy = ndimage.convolve(im, Ky)
    
    # 計算梯度的強度 (Magnitude) 與方向 (Direction)
    gradient_magnitude = np.hypot(Ix, Iy)
    gradient_direction = np.arctan2(Iy, Ix)
    
    return (gradient_magnitude, gradient_direction)

def structure_tensor(gradient_magnitude, gradient_direction, k, *args):
    # 若 hw2.py 有傳入 sigma，則取用以平滑化矩陣元素
    sigma = args[0] if len(args) > 0 else 1
    
    # 將極座標 (magnitude, direction) 轉回 Ix 與 Iy
    Ix = gradient_magnitude * np.cos(gradient_direction)
    Iy = gradient_magnitude * np.sin(gradient_direction)
    
    # 計算 M 矩陣內的元素
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # 計算 Window 內的加權總和 (使用 Gaussian filter 達到 sum_window 的效果)
    Sxx = ndimage.gaussian_filter(Ix2, sigma=sigma)
    Syy = ndimage.gaussian_filter(Iy2, sigma=sigma)
    Sxy = ndimage.gaussian_filter(Ixy, sigma=sigma)
    
    # 計算 Det(M) 與 Trace(M)
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    
    # 計算角點響應 R = det(M) - k * (trace(M))^2
    StructureTensor = det_M - k * (trace_M ** 2)
    
    return StructureTensor

def NMS(harrisim, window_size=30, threshold=0.1):
    # 找出 window_size 內的局部最大值
    local_max = ndimage.maximum_filter(harrisim, size=window_size)
    
    # 定義 Threshold 條件 (大於給定門檻值 * 圖片中的最大響應值)
    max_response = np.max(harrisim)
    
    # 角點條件：等於局部最大值，且響應強度超過門檻
    corner_mask = (harrisim == local_max) & (harrisim > threshold * max_response)
    
    # 將布林遮罩轉換回座標列表
    filtered_coords = np.argwhere(corner_mask).tolist()
    
    return filtered_coords

def plot_harris_points(image, filtered_coords):
    plt.figure()
    plt.gray()
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0]for p in filtered_coords],'+')
    plt.axis('off')
    plt.show()
    
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated