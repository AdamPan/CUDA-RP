#ifndef _OUTPUT_STYLES_H_
#define _OUTPUT_STYLES_H_

#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <sstream>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#include "SphDataType.h"
#include "bubble_CUDA.h"

extern "C" int initialize_folders();
extern "C" void *save_sph(void *threadArg);
extern "C" void *save_ascii(void *threadArg);

#endif
