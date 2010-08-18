# Source Files
EXECUTABLE 	:= bubbles
# CUDA source files
CUFILES		:= src/bubble_CUDA.cu 
#src/fortran_wrappers.cu
# CUDA dependencies
CU_DEPS     	:= inc/bubble_CUDA_kernel.cuh
# C/C++ source files (gcc 4.3)
SRCDIR		:= src/
CCFILES		:= bubbles.cpp Chameleon.cpp ConfigFile.cpp EasyBMP.cpp EasyBMP_SimpleArray.cpp

INCLUDES	+= -Iinc/

# Support multiple architectures
# Compute 1.3
# GENCODE_ARCH    += -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13
# Compute 2.0 (Fermi)
GENCODE_ARCH	+= -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20
# Debug mode
dbg		:= 0
# Fast math functions
fastmath	:= 0
# Keep ptx
keep		:= 0
# Default max registers per kernel
maxregisters	:= 32

# Rules / Targets
include ./common.mk
