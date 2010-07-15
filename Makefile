# Source Files
EXECUTABLE 	:= bubbles
# CUDA source files
CUFILES		:= src/bubble_CUDA.cu
# CUDA dependencies
CU_DEPS     	:= inc/bubble_CUDA_kernel.cuh
# inc/cuPrintf.cuh
# C/C++ source files (gcc 4.3)
SRCDIR		:= src/
CCFILES		:= bubbles.cpp Chameleon.cpp ConfigFile.cpp
INCLUDES	+= -Iinc/
# Support multiple architectures
#GENCODE_ARCH    += -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13
GENCODE_ARCH	+= -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20


#USEGLLIB	:= 1
#USEGLUT		:= 1

dbg		:= 1

# Rules / Targets
include ./common.mk
