"""
Python interface to the NVIDIA cuDNN library
"""

import sys
import ctypes
import ctypes.util
import numpy as np 
import pycuda.autoinit
import cv2

from pycuda import gpuarray
import pycuda.driver as drv



if sys.platform in ('linux2', 'linux'):
    # _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.6.5', 'libcudnn.so.6.5.18']
    _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.7', 'libcudnn.so.7.1.4']
elif sys.platform == 'win32':
    _libcudnn_libname_list = ['cudnn64_65.dll']
else:
    raise RuntimeError('unsupported platform')

_libcudnn = None
for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break
if _libcudnn is None:
    raise OSError('cuDNN library not found')

# Generic cuDNN error
class cudnnError(Exception):
    """cuDNN Error"""
    pass

class cudnnNotInitialized(cudnnError):
    """cuDNN library not initialized"""
    pass

class cudnnAllocFailed(cudnnError):
    """cuDNN allocation failed"""
    pass

class cudnnBadParam(cudnnError):
    """An incorrect value or parameter was passed to the function"""
    pass

class cudnnInvalidValue(cudnnError):
    """Invalid value"""
    pass

class cudnnArchMismatch(cudnnError):
    """Function requires an architectural feature absent from the device"""
    pass

class cudnnMappingError(cudnnError):
    """Access to GPU memory space failed"""
    pass

class cudnnExecutionFailed(cudnnError):
    """GPU program failed to execute"""
    pass

class cudnnInternalError(cudnnError):
    """An internal cudnn operation failed"""
    pass

class cudnnStatusNotSupported(cudnnError):
    """The functionality requested is not presently supported by cudnn"""
    pass

class cudnnStatusLicenseError(cudnnError):
    """License invalid or not found"""
    pass

cudnnExceptions = {
    1: cudnnNotInitialized,
    2: cudnnAllocFailed,
    3: cudnnBadParam,
    4: cudnnInternalError,
    5: cudnnInvalidValue,
    6: cudnnArchMismatch,
    7: cudnnMappingError,
    8: cudnnExecutionFailed,
    9: cudnnStatusNotSupported,
    10: cudnnStatusLicenseError
}

# Data layout specification
# cudnnTensorFormat_t is an enumerated type used by
# cudnnSetTensor4dDescriptor() to create a tensor with a pre-defined layout.
cudnnTensorFormat = {
     'CUDNN_TENSOR_NCHW': 0, # This tensor format specifies that the data is laid
                             # out in the following order: image, features map,
                             # rows, columns. The strides are implicitly defined
                             # in such a way that the data are contiguous in
                             # memory with no padding between images, feature
                             # maps, rows, and columns; the columns are the
                             # inner dimension and the images are the outermost
                             # dimension.
     'CUDNN_TENSOR_NHWC': 1 # This tensor format specifies that the data is laid
                            # out in the following order: image, rows, columns,
                            # features maps. The strides are implicitly defined in
                            # such a way that the data are contiguous in memory
                            # with no padding between images, rows, columns,
                            # and features maps; the feature maps are the
                            # inner dimension and the images are the outermost
                            # dimension.
}

# Data type
# cudnnDataType_t is an enumerated type indicating the data type to which a tensor
# descriptor or filter descriptor refers.
cudnnDataType = {
    'CUDNN_DATA_FLOAT': 0,  # The data is 32-bit single-precision floating point
                            # ( float ).
    'CUDNN_DATA_DOUBLE': 1  # The data is 64-bit double-precision floating point
                            # ( double ).
}

# cudnnAddMode_t is an enumerated type used by cudnnAddTensor4d() to specify how
# a bias tensor is added to an input/output tensor.
cudnnAddMode = {
   'CUDNN_ADD_IMAGE': 0,
   'CUDNN_ADD_SAME_HW': 0,  # In this mode, the bias tensor is defined as one
                            # image with one feature map. This image will be
                            # added to every feature map of every image of the
                            # input/output tensor.
   'CUDNN_ADD_FEATURE_MAP': 1,
   'CUDNN_ADD_SAME_CHW': 1, # In this mode, the bias tensor is defined as one
                            # image with multiple feature maps. This image
                            # will be added to every image of the input/output
                            # tensor.
   'CUDNN_ADD_SAME_C': 2,   # In this mode, the bias tensor is defined as one
                            # image with multiple feature maps of dimension
                            # 1x1; it can be seen as an vector of feature maps.
                            # Each feature map of the bias tensor will be added
                            # to the corresponding feature map of all height-by-
                            # width pixels of every image of the input/output
                            # tensor.
   'CUDNN_ADD_FULL_TENSOR': 3 # In this mode, the bias tensor has the same
                            # dimensions as the input/output tensor. It will be
                            # added point-wise to the input/output tensor.
}

# cudnnConvolutionMode_t is an enumerated type used by
# cudnnSetConvolutionDescriptor() to configure a convolution descriptor. The
# filter used for the convolution can be applied in two different ways, corresponding
# mathematically to a convolution or to a cross-correlation. (A cross-correlation is
# equivalent to a convolution with its filter rotated by 180 degrees.)
cudnnConvolutionMode = {
    'CUDNN_CONVOLUTION': 0,
    'CUDNN_CROSS_CORRELATION': 1
}

# cudnnConvolutionPath_t is an enumerated type used by the helper routine
# cudnnGetOutputTensor4dDim() to select the results to output.
cudnnConvolutionPath = {
    'CUDNN_CONVOLUTION_FORWARD': 0, # cudnnGetOutputTensor4dDim() will return
                                    # dimensions related to the output tensor of the
                                    # forward convolution.
    'CUDNN_CONVOLUTION_WEIGHT_GRAD': 1, # cudnnGetOutputTensor4dDim() will return the
                                    # dimensions of the output filter produced while
                                    # computing the gradients, which is part of the
                                    # backward convolution.
    'CUDNN_CONVOLUTION_DATA_PATH': 2 # cudnnGetOutputTensor4dDim() will return the
                                    # dimensions of the output tensor produced while
                                    # computing the gradients, which is part of the
                                    # backward convolution.
}

# cudnnAccumulateResult_t is an enumerated type used by
# cudnnConvolutionForward() , cudnnConvolutionBackwardFilter() and
# cudnnConvolutionBackwardData() to specify whether those routines accumulate
# their results with the output tensor or simply write them to it, overwriting the previous
# value.
cudnnAccumulateResults = {
    'CUDNN_RESULT_ACCUMULATE': 0,   # The results are accumulated with (added to the
                                    # previous value of) the output tensor.
    'CUDNN_RESULT_NO_ACCUMULATE': 1 # The results overwrite the output tensor.
}

# cudnnSoftmaxAlgorithm_t is used to select an implementation of the softmax
# function used in cudnnSoftmaxForward() and cudnnSoftmaxBackward() .
cudnnSoftmaxAlgorithm = {
    'CUDNN_SOFTMAX_FAST': 0,    # This implementation applies the straightforward
                                # softmax operation.
    'CUDNN_SOFTMAX_ACCURATE': 1 # This implementation applies a scaling to the input
                                # to avoid any potential overflow.
}

# cudnnSoftmaxMode_t is used to select over which data the cudnnSoftmaxForward()
# and cudnnSoftmaxBackward() are computing their results.
cudnnSoftmaxMode = {
    'CUDNN_SOFTMAX_MODE_INSTANCE': 0,   # The softmax operation is computed per image (N)
                                        # across the dimensions C,H,W.
    'CUDNN_SOFTMAX_MODE_CHANNEL': 1     # The softmax operation is computed per spatial
                                        # location (H,W) per image (N) across the dimension
                                        # C.
}

# cudnnPoolingMode_t is an enumerated type passed to
# cudnnSetPoolingDescriptor() to select the pooling method to be used by
# cudnnPoolingForward() and cudnnPoolingBackward() .
cudnnPoolingMode = {
    'CUDNN_POOLING_MAX': 0,     # The maximum value inside the pooling window will
                                # be used.
    'CUDNN_POOLING_AVERAGE': 1  # The values inside the pooling window will be
                                # averaged.
}

# cudnnActivationMode_t is an enumerated type used to select the neuron activation
# function used in cudnnActivationForward() and cudnnActivationBackward() .
cudnnActivationMode = {
    'CUDNN_ACTIVATION_SIGMOID': 0,  # sigmoid function
    'CUDNN_ACTIVATION_RELU': 1,     # rectified linear function
    'CUDNN_ACTIVATION_TANH': 2      # hyperbolic tangent function
}

def cudnnCheckStatus(status):
    """
    Raise cuDNN exception

    Raise an exception corresponding to the specified cuDNN error code.

    Parameters
    ----------
    status : int
        cuDNN error code
    """

    if status != 0:
        try:
            raise cudnnExceptions[status]
        except KeyError:
            raise cudnnError



# Helper functions
_libcudnn.cudnnCreate.restype = int
_libcudnn.cudnnCreate.argtypes = [ctypes.c_void_p]
def cudnnCreate():
    """
    Initialize cuDNN.

    Initializes cuDNN and returns a handle to the cuDNN context.

    Returns
    -------

    handle : cudnnHandle
        cuDNN context
    """

    handle = ctypes.c_void_p()
    status = _libcudnn.cudnnCreate(ctypes.byref(handle))
    cudnnCheckStatus(status)
    return handle.value


_libcudnn.cudnnDestroy.restype = int
_libcudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]
def cudnnDestroy(handle):
    """
    Release cuDNN resources.

    Release hardware resources used by cuDNN.

    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    """

    status = _libcudnn.cudnnDestroy(ctypes.c_void_p(handle))
    cudnnCheckStatus(status)



_libcudnn.cudnnCreateTensorDescriptor.restype = int
_libcudnn.cudnnCreateTensorDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateTensorDescriptor():
    """
    Create a Tensor descriptor object.

    Allocates a cudnnTensor4dDescriptor_t structure and returns a pointer to it.

    Returns
    -------
    tensor4d_descriptor : int
        Tensor4d descriptor.
    """

    tensor = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(tensor))
    cudnnCheckStatus(status)
    return tensor.value

_libcudnn.cudnnSetTensor4dDescriptor.restype = int
_libcudnn.cudnnSetTensor4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]
def cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w):
    """
    Initialize a previously created Tensor 4D object.

    This function initializes a previously created Tensor4D descriptor object. The strides of
    the four dimensions are inferred from the format parameter and set in such a way that
    the data is contiguous in memory with no padding between dimensions.

    Parameters
    ----------
    tensorDesc : cudnnTensor4dDescriptor
        Handle to a previously created tensor descriptor.
    format : cudnnTensorFormat
        Type of format.
    dataType : cudnnDataType
        Data type.
    n : int
        Number of images.
    c : int
        Number of feature maps per image.
    h : int
        Height of each feature map.
    w : int
        Width of each feature map.
    """

    status = _libcudnn.cudnnSetTensor4dDescriptor(tensorDesc, format, dataType,
                                                  n, c, h, w)
    cudnnCheckStatus(status)




_libcudnn.cudnnCreateFilterDescriptor.restype = int
_libcudnn.cudnnCreateFilterDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateFilterDescriptor():
    """"
    Create a filter descriptor.
    
    This function creates a filter descriptor object by allocating the memory needed to hold
its opaque structure.
    
    Parameters
    ----------

    
    Returns
    -------
    filterDesc : cudnnFilterDescriptor
        Handle to a newly allocated filter descriptor.
    """

    filterDesc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateFilterDescriptor(ctypes.byref(filterDesc))
    cudnnCheckStatus(status)
    
    return filterDesc.value


_libcudnn.cudnnSetFilter4dDescriptor.restype = int
_libcudnn.cudnnSetFilter4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w):
    
    status = _libcudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w)
    cudnnCheckStatus(status)





_libcudnn.cudnnCreateConvolutionDescriptor.restype = int
_libcudnn.cudnnCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateConvolutionDescriptor():
    
    convDesc = ctypes.c_void_p()
    
    status = _libcudnn.cudnnCreateConvolutionDescriptor(ctypes.byref(convDesc))
    cudnnCheckStatus(status)
    
    return convDesc.value






_libcudnn.cudnnSetConvolution2dDescriptor.restype = int
_libcudnn.cudnnSetConvolution2dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

def cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, v_stride, h_stride, d_h, d_w, mode, computeType):
    
    status = _libcudnn.cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, v_stride, h_stride, d_h, d_w, mode, computeType)
    cudnnCheckStatus(status)






_libcudnn.cudnnGetConvolutionForwardAlgorithm.restype = int
_libcudnn.cudnnGetConvolutionForwardAlgorithm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

def cudnnGetConvolutionForwardAlgorithm(handle, input_desc, kernel_desc, convolution_desc, out_desc, conv_fwd_pref, memory_limit_bytes):

    forward_convolution_algo = ctypes.c_void_p()

    status = _libcudnn.cudnnGetConvolutionForwardAlgorithm(handle, input_desc, kernel_desc, convolution_desc, out_desc, conv_fwd_pref, memory_limit_bytes, ctypes.byref(forward_convolution_algo))
    cudnnCheckStatus(status)


    return forward_convolution_algo.value




_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

def cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, kernel_desc, convolution_desc, out_desc, convolution_algo ):

    workspace_bytes = ctypes.c_void_p()

    status = _libcudnn.cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, kernel_desc, convolution_desc, out_desc, convolution_algo, ctypes.byref(workspace_bytes))
    cudnnCheckStatus(status)


    return workspace_bytes.value



_libcudnn.cudnnConvolutionForward.restype = int
_libcudnn.cudnnConvolutionForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnConvolutionForward(handle, alpha, input_desc, d_input, kernel_desc, d_kernel, conv_desc, conv_algo, d_workspace, workspace_bytes, beta, out_desc, d_output):

    _alpha_ = ctypes.c_float(alpha)
    _beta_ = ctypes.c_float(beta)

    status = _libcudnn.cudnnConvolutionForward(handle, ctypes.byref(_alpha_), input_desc, d_input, kernel_desc, d_kernel, conv_desc, 
                                                conv_algo, d_workspace, workspace_bytes, ctypes.byref(_beta_), out_desc, d_output)


    cudnnCheckStatus(status)
    return d_output.value

