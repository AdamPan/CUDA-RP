/*
 * A simple test of using CUDA to solve many Rayleigh Plesset problems in parallel
 * 
 * Written by Adam Pan
 *
 */

#ifndef _BUBBLES_H_
#define _BUBBLES_H_

#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <sstream>

#include "Chameleon.h"

#include <time.h>

class ConfigFile{
	std::map<std::string,Chameleon> content_;

public:
	ConfigFile(std::string const& configFile);

	Chameleon const& Value(std::string const& section, std::string const& entry) const;

	Chameleon const& Value(std::string const& section, std::string const& entry, double value);
	Chameleon const& Value(std::string const& section, std::string const& entry, std::string const& value);
};

#endif
