#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

Ptr<CvRTrees> TrainClassifier(const Mat& trainData, const Mat& trainResponses, int trsum, int tesum, int vocsize);

#endif CLASSIFIER_H