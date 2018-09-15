
from cudnnlib import *
import cv2
import numpy as np

def main():
    cudnn_handle = cudnnCreate()
    
    input_tensor_descriptor = cudnnCreateTensorDescriptor()
    cudnnSetTensor4dDescriptor(input_tensor_descriptor, 1, 0, 1, 3, 1170, 1920)

    output_tensor_descriptor = cudnnCreateTensorDescriptor()
    cudnnSetTensor4dDescriptor(output_tensor_descriptor, 1, 0, 1, 3, 1170, 1920)

    kernel_descriptor = cudnnCreateFilterDescriptor()

    cudnnSetFilter4dDescriptor(kernel_descriptor, 0, 0, 3, 3, 3, 3)

    convolution_descriptor = cudnnCreateConvolutionDescriptor()
    cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, 1, 0)

    convolution_algo = cudnnGetConvolutionForwardAlgorithm(cudnn_handle, input_tensor_descriptor, kernel_descriptor, convolution_descriptor, output_tensor_descriptor, 1, 0)

    workspace_bytes = cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_tensor_descriptor, kernel_descriptor, convolution_descriptor, output_tensor_descriptor, convolution_algo)
    
    print("workspace size : ",  workspace_bytes/1024/1024 , " MB")


    d_workspace = drv.mem_alloc(workspace_bytes)

    d_workspace_data = ctypes.c_void_p(int(d_workspace))



    img = cv2.imread("baby.jpg", 1)
    img = np.float32(img) 
    img /= np.max(img)

    rows, columns, depth = img.shape

    image_bytes = 1 * depth * rows * columns * 4


    d_input = drv.mem_alloc(image_bytes)

    drv.memcpy_htod(d_input, img.astype("float32"))

    d_input_data = ctypes.c_void_p(int(d_input))


    # temp_arr = np.zeros(img.shape, np.float32)
    # drv.memcpy_dtoh(temp_arr, d_input_data.value )
    # temp_arr = np.uint8(temp_arr)
    # cv2.imwrite("temp_img.png", temp_arr)

    
    d_output = drv.mem_alloc(image_bytes)

    d_output_data = ctypes.c_void_p(int(d_output))


    kernel_template = np.zeros((3,3), np.float32)

    kernel_template[:] = 1.0

    kernel_template[1,1] = -8.0

    h_kernel = np.zeros((3,3,3,3), np.float32)

    for kernel in range(3):
        for channel in range(3):
            for row_ in range(3):
                for column_ in range(3):
                    h_kernel[kernel][channel][row_][column_] = kernel_template[row_][column_]


    d_kernel = drv.mem_alloc(h_kernel.nbytes)

    drv.memcpy_htod(d_kernel, h_kernel)

    d_kernel_data = ctypes.c_void_p(int(d_kernel))


    alpha = 1.0
    beta = 0.0

    new_output = cudnnConvolutionForward(cudnn_handle, alpha, input_tensor_descriptor, d_input_data, kernel_descriptor, d_kernel_data, convolution_descriptor, convolution_algo, d_workspace_data, workspace_bytes, beta, output_tensor_descriptor, d_output_data)

    result_arr = np.zeros(img.shape, np.float32)
    drv.memcpy_dtoh(result_arr, new_output)

    # print(np.min(result_arr), np.max(result_arr), len(np.unique(result_arr)))


    result_arr[result_arr<=0] = 0

    result_arr /= np.max(result_arr)

    result_arr = np.uint8(result_arr * 255)
    print(result_arr.dtype, result_arr.shape)

    cv2.imwrite("result.png", result_arr)
    cudnnDestroy(cudnn_handle)
    print("done")


if __name__ == '__main__':
    main()