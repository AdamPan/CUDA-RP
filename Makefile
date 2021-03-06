# Source Files
EXECUTABLE 	:= bubbles
# CUDA source files
CUFILES		:= src/bubble_CUDA.cu 
#src/fortran_wrappers.cu
# CUDA dependencies
CU_DEPS     	:= inc/bubble_CUDA_kernel.cuh inc/double_vector_math.cuh inc/thrust_tuples.cuh
# C/C++ source files (gcc 4.3)
SRCDIR		:= src/
CCFILES		:= bubbles.cpp SphDataType.cpp output_styles.cpp

INCLUDES	+= -Iinc/

# Support multiple architectures
# Compute 1.3
GENCODE_ARCH    += -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13
# Compute 2.0 (Fermi)
GENCODE_ARCH	+= -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20

USEBOOST		:= 1

# Default max registers per kernel
maxregisters	:= 32

# Debug mode
dbg		:= 0
# Keep ptx
keep		:= 0


# Rules / Targets
include ./common.mk
