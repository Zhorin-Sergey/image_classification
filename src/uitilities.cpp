#include "utilities.h"

using namespace std;

void GetFilesInFolder(const string& dirPath, std::vector<string> &filesList, int &count)
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

void getHOGfeatures(const string& fileName, vector<float>  descriptors)
{
 
  Size winSize, blockSize = Size(16, 16), winStride, padding;
  Size blockStride = Size(8, 8), cellSize = Size(8, 8);
  Size imgSize = Size(128, 96);
  int nbins = 9;
  Mat img, resizedImg;

  img = imread(fileName, -1);
  if (img.cols < img.rows)
  {
    imgSize.height = 128;
    imgSize.width = 96;
  }
  resize(img, resizedImg, imgSize);

  winSize = Size(resizedImg.cols, resizedImg.rows);
  HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
  hog.compute(resizedImg, descriptors);
  img.release();
  resizedImg.release();

 /* for (i = 0; i < descriptors.size(); i++)
  {
    fprintf(outFile, ";%lf", descriptors[i]);
  }
  fprintf(outFile, "\n");
  */
}
