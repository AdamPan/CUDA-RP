#ifndef _BUBBLES_H_
#define _BUBBLES_H_

#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <sstream>

#include "ConfigFile.h"

#include <time.h>

int runSimulation(int argc, char **argv);

extern "C" int initialize_folders();

extern "C" void *save_step(void *threadArg);

#endif
