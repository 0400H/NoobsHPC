# Copyright (c) 2018 NoobsDNN Authors, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(ENABLE_VERBOSE_MSG)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

if(DISABLE_ALL_WARNINGS)
    noobsdnn_disable_warnings(CMAKE_CXX_FLAGS)
endif()

# find cuda
if(USE_CUDA)
    #set other cuda path
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
    noobsdnn_find_cuda()
endif()

# find cudnn default cudnn 5
if(USE_CUDNN)
    noobsdnn_find_cudnn()
endif()

# find opencl
if(USE_OPENCL)
    noobsdnn_generate_kernel(${NBDNN_ROOT})
    noobsdnn_find_opencl()
endif()

if(BUILD_WITH_GLOG)
    noobsdnn_find_glog()
endif()

if(USE_PROTOBUF)
    noobsdnn_find_protobuf()
    noobsdnn_protos_processing()
endif()

if (USE_GFLAGS)
    noobsdnn_find_gflags()
endif()

if(USE_MKL)
    #noobsdnn_find_mkl()
endif()

if (USE_XBYAK)
    #noobsdnn_find_xbyak()
endif()
if (USE_MKLML)
    #noobsdnn_find_mklml()
endif()
