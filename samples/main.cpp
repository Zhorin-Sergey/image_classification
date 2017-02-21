#include <list>

#include <opencv2/opencv.hpp>

#include "utilities.h"
#include "bow.h"
#include "Classifier.h"

using namespace cv;

/*#define train1 "D:/games/Reposit/Cv/image_classification_build/samples/train/indoor-dwelling-train"
#define train2 "D:/games/Reposit/Cv/image_classification_build/samples/train/indoor-msu-train"
#define train3 "D:/games/Reposit/Cv/image_classification_build/samples/train/outdoor-blossoms-train"
#define train4 "D:/games/Reposit/Cv/image_classification_build/samples/train/outdoor-urban-train"
#define test1 "D:/games/Reposit/Cv/image_classification_build/samples/test/indoor-dwelling-test" 
#define test2 "D:/games/Reposit/Cv/image_classification_build/samples/test/indoor-msu-test" 
#define test3 "D:/games/Reposit/Cv/image_classification_build/samples/test/outdoor-blossoms-test"
#define test4 "D:/games/Reposit/Cv/image_classification_build/samples/test/outdoor-urban-test"
//#define vocsize 50*/


int main(int argc, char* argv[])
{
  int te1 = 0, te2 = 0, te3 = 0, te4 = 0, tr1 = 0, tr2 = 0, tr3 = 0, tr4 = 0;
  std::vector<string> filesList;
  GetFilesInFolder(argv[1], filesList, tr1);
  GetFilesInFolder(argv[2], filesList, tr2);
  GetFilesInFolder(argv[3], filesList, tr3);
  GetFilesInFolder(argv[4], filesList, tr4);
  GetFilesInFolder(argv[5], filesList, te1);
  GetFilesInFolder(argv[6], filesList, te2);
  GetFilesInFolder(argv[7], filesList, te3);
  GetFilesInFolder(argv[8], filesList, te4);
  int vocsize = atoi(argv[9]);

  int sum = tr1 + te1 + te2 + tr2 + te3 + tr3 + te4 + tr4;
  int trsum = tr1 + tr2  + tr3 + tr4;
  int tesum = te1 + te2 +  te3 + te4;
  std::vector<Mat> descriptors(sum);
  std::vector<Mat> imgDesc;
 // Mat samples(sum, vocsize, CV_32F);
  Mat testSamples(tesum, vocsize, CV_32F);
  Mat trainSamples(trsum, vocsize, CV_32F);
  std::vector<std::vector<KeyPoint>> keypoints(sum);
  int i = 0;
  
  Mat voc;
  while (i < sum)
  {
    DetectKeypointsOnImage(filesList[i], keypoints[i], descriptors[i], argv[10], argv[11]);
    
    i++;
  }
  printf("DetectKeypointsOnImage\n");
  i = 0;
  voc = BuildVocabulary(descriptors, vocsize, tr1 + tr2 + tr3 + tr4);
  printf("BuildVocabulary\n");
  while (i < trsum)
  {
    ComputeImgDescriptor(filesList[i], voc, trainSamples.row(i), argv[10], argv[11]);
    //getHOGfeatures(filesList[i], trainSamples.row(i));
    i++;
  }
  int j = 0;
  while (i < sum)
  {
    ComputeImgDescriptor(filesList[i], voc, testSamples.row(j), argv[10], argv[11]);
   // getHOGfeatures(filesList[i], testSamples.row(j));
    i++;
    j++;
  }
  i = 0;
  while (i < trsum*vocsize)
  {
    trainSamples.at<int>(i) = 0;

    i++;
  }
  printf("ComputeImgDescriptor\n");
 // Mat labels(sum, 1, CV_32S);
  Mat trainLabels(trsum, 1, CV_32S);
  Mat testLabels(tesum, 1, CV_32S);
  int ind = 0;
  for (int j = 0; j < tr1; j++, ind++)
    trainLabels.at<int>(ind) = 1;
  for (int j = 0; j < tr2; j++, ind++)
    trainLabels.at<int>(ind) = 2;
  for (int j = 0; j < tr3; j++, ind++)
    trainLabels.at<int>(ind) = 3;
  for (int j = 0; j < tr4; j++, ind++)
    trainLabels.at<int>(ind) = 4;
  ind = 0;
  for (int j = 0; j < te1; j++, ind++)
    testLabels.at<int>(ind) = 1;
  for (int j = 0; j < te2; j++, ind++)
    testLabels.at<int>(ind) = 2;
  for (int j = 0; j < te3; j++, ind++)
    testLabels.at<int>(ind) = 3;
  for (int j = 0; j < te4; j++, ind++)
    testLabels.at<int>(ind) = 4;
 // CvDTree rf;
 // Ptr<CvRTrees> rtp = &rf;
  printf("labels\n");

  /*Mat trainSampleMask(1, trsum, CV_32S);
  for (int i = 0; i < trsum; ++i) {
    trainSampleMask.at<int>(i) = i;
  }*/
  /* for (int i = trsum; i < tesum + trsum; ++i) {
  trainSampleMask.at<int>(i) = 0;
  }*/

  CvGBTreesParams params = CvGBTreesParams();
  params.max_depth = 3; 
 // params.min_sample_count = 1;
  params.weak_count = atoi(argv[12]);
  params.shrinkage = 0.5f; 
  params.subsample_portion = 1.0f; 
  params.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
  // CvRTParams params = CvRTParams();
 // CvDTreeParams params = CvDTreeParams();
 // params.cv_folds = 5;
 // params.max_depth = atoi(argv[12]);
  //params.min_sample_count = 1000;
 // params.calc_var_importance = false;
  //params.term_crit.type = CV_TERMCRIT_ITER;
  printf("PointOfNoReturn\n");
  //params.term_crit.max_iter = atoi(argv[12]);
  printf("PointOfNoReturn\n");
  //CvRTrees rf;
  
 // Ptr<CvRTrees> rf1 = &rf;
  Mat varIdx(1, vocsize, CV_8U, Scalar(1));
 // varIdx.at<int>(vocsize) = 0;
  Mat varTypes(1, vocsize + 1, CV_8U, Scalar(CV_VAR_ORDERED));
  varTypes.at<uchar>(vocsize) = CV_VAR_CATEGORICAL;
  printf("PointOfNoReturn\n"); 
  CvGBTrees rf = CvGBTrees(trainSamples, CV_ROW_SAMPLE, trainLabels, Mat(), Mat(), varTypes, Mat(), params);
  rf.train(trainSamples, CV_ROW_SAMPLE, trainLabels, Mat(), Mat(), varTypes, Mat(), params);
  

  //TrainClassifier(samples, labels, trsum, tesum, vocsize, rtp, atoi(argv[12]), atoi(argv[13]));
  printf("TrainClassifier\n");
  rf.save("model-rf.yml", "simpleRTreesModel");
  printf("save\n");
  float trainError = 0.0f;
  for (int i = 0; i < trsum; ++i) {
    int prediction = (int)(rf.predict(trainSamples.row(i)));
    printf("%i ", prediction);
    trainError += (trainLabels.at<int>(i) != prediction);
  }
  printf("TEST\n");
  trainError /= float(trsum);
  float testError = 0.0f; 
  for(int i = 0; i <tesum; ++i) {
    int prediction = (int)(rf.predict(testSamples.row(i)));
    printf("%i ", prediction);
    testError += (testLabels.at<int>(i) != prediction);
  }
  testError /= float(sum - trsum);
  printf("train error = %.4f\ntest error = %.4f\n", trainError, testError);
  return 0;

}