#include "Classifier.h"

void TrainClassifier(const Mat& trainData, const Mat& trainResponses, int trsum, int tesum, int vocsize, Ptr<CvRTrees> rtp) {
  Mat trainSampleMask(1, trsum, CV_32S);
  for (int i = 0; i < trsum; ++i) {
    trainSampleMask.at<int>(i) = i;
  }
 /* for (int i = trsum; i < tesum + trsum; ++i) {
    trainSampleMask.at<int>(i) = 0;
  }*/
  CvRTParams params; 
  params.max_depth = 10;
  params.min_sample_count = 1; 
  params.calc_var_importance = false; 
  params.term_crit.type = CV_TERMCRIT_ITER;
  params.term_crit.max_iter = 250;
  CvRTrees rf;
  Ptr<CvRTrees> rf1 = &rf;
  Mat varIdx(1, vocsize, CV_8U, Scalar(1));
  //varIdx.at<int>(vocsize) = 0;
  Mat varTypes(1, vocsize + 1, CV_8U, Scalar(CV_VAR_ORDERED));
  varTypes.at<uchar>(vocsize) = CV_VAR_CATEGORICAL; 
  printf("PointOfNoReturn\n");
  rtp->train(trainData, CV_ROW_SAMPLE, trainResponses, varIdx, trainSampleMask, varTypes, Mat(), params);
  printf("PointOfNoReturn\n");

}