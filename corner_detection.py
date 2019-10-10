# Kesar TN
# University of Central Florida
# kesar@Knights.ucf.edu

import numpy as np
from PIL import Image

# Function to convolve the matrix
def convolution(image, kernel):
    
    # Extract image and kernel shape
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    # Create the output image shape
    out = np.zeros(shape=image.shape)
    
    # Create a padded image with extra pixels for convolution and place the image in the center
    image_padded = np.zeros(shape=(height + kernel_height, width + kernel_width))
    image_padded[kernel_height//2 : -kernel_height//2, kernel_width//2 : -kernel_width//2] = image
    
    # Iterate through each and every value of both the image and the kernel
    for row in range(height):
        for col in range(width):
            for i in range(kernel_height):
                for j in range(kernel_width):
                    # Multiply kernel values and store in the output
                    out[row, col] += image_padded[row + i, col + j] * kernel[i, j]

    return out

# Function for Sobel edge detection
def harris_corner_detector(image, window_size, k, threshold):
    # Open images using PIL
    image = Image.open(image)
    image = np.asarray(image)
    # Define a window size and an empty array for output
    window = np.ones((window_size, window_size))
    out = np.zeros_like(image)

    # Define sobel x and y kernels
    sobel_kernelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Convolve these kernels with the input image
    print('Detecting Sobel edges..')
    x_derivative = convolution(image, sobel_kernelx)
    y_derivative = convolution(image, sobel_kernely)
    
    # Find Ixx, Ixy and Iyy
    Ixx = x_derivative * x_derivative
    Ixy = x_derivative * y_derivative    
    Iyy = y_derivative * y_derivative  
    
    # Using the window functions, convolve the images
    print('Using window function..')
    Sxx = convolution(Ixx, window)
    Sxy = convolution(Ixy, window)
    Syy = convolution(Iyy, window)

    # Iterating through each pixel
    print('Harris detection..')
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            # calculate Hxy for the entire image and then R(x, y)
            Hxy = [[Sxx[i,j], Sxy[i,j]], [Sxy[i,j], Syy[i,j]]]
            R = np.linalg.det(Hxy) - (k * np.trace(Hxy)**2)

            # Using the defined threshold, place white dots
            if R > threshold:
                out[i, j] = 255

    # Display the output and then save it
    out = Image.fromarray(out)
    out.save('harris_corners_2.png')
    out.show()

if __name__=='__main__':
    
    # Main function for Harris corner detection
    harris_corner_detector('input_hcd2.jpg', 5, 0.06, 3e9)