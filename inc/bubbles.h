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
#include <iterator>

#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

#include "bubble_CUDA.h"
#include "output_styles.h"

enum
{
	none,
	ascii,
	sph,
};


int runSimulation(int argc, char **argv);

#endif
