# HW2: Harris Corner Detection

**Institute of Artificial Intelligence Innovation, National Yang Ming Chiao Tung University** **Course:** Computer Vision (IIAI30013)

---

## 1. Required Libraries
The following libraries are used in this assignment. Please ensure they are installed in your Python environment before execution:
* `os`
* `cv2` (opencv-python)
* `numpy`
* `matplotlib`
* `scipy`

> **Notice:** `cv2` is only used for basic image I/O and conversion (e.g., BGR2RGB, BGR2GRAY, resizing, and rotation affine transforms). It is **not** used directly for the required core algorithms (i.e., no `cv2.Sobel`, `cv2.Laplacian`, or `cv2.cornerHarris` are used).

## 2. Python Files
* `hw2.py`: The main execution script that loads the image, calls the algorithms, and saves the output results.
* `Harris_Corner_Detection.py`: The function file containing the pure implementations of the Harris Corner Detection pipeline.

## 3. Execution Instructions
Before running the program, please make sure the input image named `original.jpg` is placed in the exact same directory as the Python scripts. 

To execute the program and generate all required images, open your terminal or command prompt and enter the following command:

```bash
python hw2.py
```

## 4. Functions Implemented
The core logic resides in `Harris_Corner_Detection.py`. The following functions were implemented from scratch:

* `gaussian_smooth(size, sigma=1)`: Creates a 2D Gaussian kernel and filters images with Gaussian blur to remove noise.
* `sobel_edge_detection(im)`: Applies Sobel filters to the blurred images and computes both the magnitude and direction of the gradients.
* `structure_tensor(gradient_magnitude, gradient_direction, k)`: Uses the gradient magnitude to compute the structure tensor (second-moment matrix) and calculates the cornerness response ($R = \det(M) - k \cdot (\text{trace}(M))^2$).
* `NMS(harrisim, window_size=30, threshold=0.1)`: Performs Non-Maximum Suppression (NMS) to find local maxima within a specified window size, filtering out weak corners based on a given threshold.
* `rotate(image, angle, center=None, scale=1.0)`: Rotates the original image by a specified angle for testing rotation invariance.

## 5. Output Results
After a successful run, all generated images will be saved in the `results` directory, categorized into five subfolders:
1. `Gaussian smooth results/` (2 images)
    * Results using $\sigma=5$ with kernel sizes $5\times5$ and $10\times10$.

2. `Sobel edge detection results/` (4 images)
    * Magnitude of gradient for kernel sizes 5 and 10.
    * Direction of gradient for kernel sizes 5 and 10.

3. `Structure tensor + NMS results/` (2 images)
    * Corner detection results using a $3\times3$ NMS window.
    * Corner detection results using a $30\times30$ NMS window.

4. `Final results of rotating/` (1 image)
    * Corner detection result after rotating the original image by 30°.

5. `Final results of scaling/` (1 image)
    * Corner detection result after scaling the original image down to 0.5x.