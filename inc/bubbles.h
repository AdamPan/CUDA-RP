#ifndef _BUBBLES_H_
#define _BUBBLES_H_

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

#include "boost/program_options/environment_iterator.hpp"
#include "boost/program_options/eof_iterator.hpp"
#include "boost/program_options/option.hpp"
#include "boost/program_options/options_description.hpp"
#include "boost/program_options/parsers.hpp"
#include "boost/program_options/positional_options.hpp"
#include "boost/program_options/value_semantic.hpp"
#include "boost/program_options/variables_map.hpp"

#include "ConfigFile.h"
#include "SphDataType.h"
#include "bubble_CUDA.h"

int runSimulation(int argc, char **argv);

extern "C" int initialize_folders();

extern "C" void *save_sph(void *threadArg);
extern "C" void *save_ascii(void *threadArg);

#endif
