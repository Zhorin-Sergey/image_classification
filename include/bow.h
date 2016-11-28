#ifndef BOW_H
#define BOW_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

void DetectKeypointsOnImage(const string& fileName, vector<KeyPoint>& keypoints, Mat& descriptors);
Mat BuildVocabulary(const std::vector<Mat>& descriptors, int vocSize, size_t n);
void ComputeImgDescriptor(const string& fileName, Mat& voc, Mat& imgDesc);

#endif
