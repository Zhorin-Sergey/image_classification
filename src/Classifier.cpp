#include "Classifier.h"

Ptr<CvRTrees> TrainClassifier(const Mat& trainData, const Mat& trainResponses, int trsum, int tesum, int vocsize) {
  Mat trainSampleMask(1, trsum + tesum, CV_8U);
  for (int i = 0; i < trsum; ++i) {
    trainSampleMask.at<int>(i) = 1;
  }
  for (int i = trsum; i < tesum + trsum; ++i) {
    trainSampleMask.at<int>(i) = 0;
  }
  CvRTParams params; 
  params.max_depth = 10;
  params.min_sample_count = 1; 
  params.calc_var_importance = false; 
  params.term_crit.type = CV_TERMCRIT_ITER;
  params.term_crit.max_iter = 250;
  CvRTrees rf;
  Mat varIdx(1, vocsize + 1, CV_8U, Scalar(1));
  varIdx.at<int>(vocsize) = 0;
  Mat varTypes(1, vocsize + 1, CV_8U, Scalar(CV_VAR_ORDERED));
  varTypes.at<uchar>(vocsize) = CV_VAR_CATEGORICAL;
  rf.train(trainData, CV_ROW_SAMPLE, trainResponses, varIdx, trainSampleMask, varTypes, Mat(), params);
  return &rf;
}