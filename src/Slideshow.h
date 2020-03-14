#ifndef Slideshow_H
#define Slideshow_H

#include "PhotoSelector.h"
#include <string>
#include <fstream>

void parseInputFile(std::string &inputFileName, PhotoSelector &photoSelector);
void parsePhoto(std::string photoLine);
void writeOutputFile(std::string &ouputFileName);

#endif //Slideshow_H
