#ifndef Slideshow_H
#define Slideshow_H

#include "PhotoSelector.h"
#include <string>
#include <fstream>

void parseInputFile(string &inputFileName, PhotoSelector &photoSelector);
void parsePhoto(string photoLine);
void writeOutputFile(string &ouputFileName, PhotoSelector &photoSelector);

#endif //Slideshow_H
