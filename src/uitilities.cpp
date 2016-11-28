#include "utilities.h"

using namespace std;

void GetFilesInFolder(const string& dirPath, std::vector<string> &filesList, int count)
{
  HANDLE handle;
  WIN32_FIND_DATAA fileData;

  if ((handle = FindFirstFileA((dirPath + "/*.png").c_str(), &fileData)) == INVALID_HANDLE_VALUE)
  {
    return;
  }
  do
  {
    count++;
    const string file_name = fileData.cFileName;
    const string full_file_name = dirPath + "/" + file_name;
    filesList.push_back(full_file_name);
  } while (FindNextFileA(handle, &fileData));
  FindClose(handle);
}

