#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <vector>
#include <windows.h>

void GetFilesInFolder(const std::string& dirPath,
    std::vector<std::string> &filesList, int &count);

#endif
