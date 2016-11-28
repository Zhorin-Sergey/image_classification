#include <list>

#include <opencv2/opencv.hpp>

#include "utilities.h"
#include "bow.h"

using namespace cv;

#define train1 "train"
#define train2 "train"
#define train3 "train"
#define train4 "train"
#define test1 "test" 
#define test2 "test" 
#define test3 "test"
#define test4 "test"
#define vocsize 100


int main()
{
  int te1 = 0, te2 = 0, te3 = 0, te4 = 0, tr1 = 0, tr2 = 0, tr3 = 0, tr4 = 0;
  std::vector<string> filesList;
  GetFilesInFolder(train1, filesList, tr1);
  GetFilesInFolder(train2, filesList, tr2);
  GetFilesInFolder(train3, filesList, tr3);
  GetFilesInFolder(train4, filesList, tr4);
  GetFilesInFolder(test1, filesList, te1);
  GetFilesInFolder(test2, filesList, te2);
  GetFilesInFolder(test3, filesList, te3);
  GetFilesInFolder(test4, filesList, te4);
  int sum = tr1 + te1 + te2 + tr2 + te3 + tr3 + te4 + tr4;
  std::vector<Mat> descriptors;
  std::vector<Mat> imgDesc;
  std::vector<std::vector<KeyPoint>> keypoints;
  int i = 0;
  Mat voc;
  while (i < sum)
  {
    DetectKeypointsOnImage(filesList[i], keypoints[i], descriptors[i]);
    i++;
  }
  i = 0;
  voc = BuildVocabulary(descriptors, vocsize, tr1 + tr2 + tr3 + tr4);
  while (i < sum)
  {
    ComputeImgDescriptor(filesList[i], voc, imgDesc[i]);
    i++;
  }
  int * aprClassInfo = new int[sum];
  int ind = 0;
  for (int j = 0 ; j < tr1; j++, ind++)
    aprClassInfo[ind] = 1;
  for (int j = 0; j < tr2; j++, ind++)
    aprClassInfo[ind] = 2;
  for (int j = 0; j < tr3; j++, ind++)
    aprClassInfo[ind] = 3;
  for (int j = 0; j < tr4; j++, ind++)
    aprClassInfo[ind] = 4;
  for (int j = 0; j < te1; j++, ind++)
    aprClassInfo[ind] = 1;
  for (int j = 0; j < te2; j++, ind++)
    aprClassInfo[ind] = 2;
  for (int j = 0; j < te3; j++, ind++)
    aprClassInfo[ind] = 3;
  for (int j = 0; j < te4; j++, ind++)
    aprClassInfo[ind] = 4;

  return 0;
}