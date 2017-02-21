#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void GetFilesInFolder(const std::string& dirPath,
    std::vector<std::string> &filesList, int &count);
void getHOGfeatures(const string& fileName, vector<float>  descriptors);

#endif
