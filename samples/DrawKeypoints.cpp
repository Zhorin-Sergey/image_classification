#include <opencv2/opencv.hpp>

#include "utilities.h"
#include "bow.h"
#include "Classifier.h"

void main(int argc, char* argv[]) {
  const char *srcWinName = "src", *KeyPoinst = "keypoints"; 
  namedWindow(srcWinName, 1); 
  namedWindow(KeyPoinst, 1);
  Mat img = imread(argv[1]);
  Mat imgOut;
  std::vector < KeyPoint > keypoints;
  Mat descriptor;
  DetectKeypointsOnImage(argv[1], keypoints, descriptor, argv[2], argv[3]);
  drawKeypoints(img, keypoints, imgOut);
  imshow(KeyPoinst, imgOut);
  imshow(srcWinName, img);
  waitKey(0);
}