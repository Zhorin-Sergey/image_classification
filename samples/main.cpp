#include <list>

#include <opencv2/opencv.hpp>

#include "utilities.h"
#include "bow.h"
#include "Classifier.h"

using namespace cv;

#define train1 "D:/games/Reposit/Cv/image_classification_build/samples/train/indoor-dwelling-train"
#define train2 "D:/games/Reposit/Cv/image_classification_build/samples/train/indoor-msu-train"
#define train3 "D:/games/Reposit/Cv/image_classification_build/samples/train/outdoor-blossoms-train"
#define train4 "D:/games/Reposit/Cv/image_classification_build/samples/train/outdoor-urban-train"
#define test1 "D:/games/Reposit/Cv/image_classification_build/samples/test/indoor-dwelling-test" 
#define test2 "D:/games/Reposit/Cv/image_classification_build/samples/test/indoor-msu-test" 
#define test3 "D:/games/Reposit/Cv/image_classification_build/samples/test/outdoor-blossoms-test"
#define test4 "D:/games/Reposit/Cv/image_classification_build/samples/test/outdoor-urban-test"
#define vocsize 50


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
  int trsum = tr1 + tr2  + tr3 + tr4;
  int tesum = te1 + te2 +  te3 + te4;
  std::vector<Mat> descriptors(sum);
  std::vector<Mat> imgDesc;
  Mat samples(sum, vocsize, CV_32F);
  std::vector<std::vector<KeyPoint>> keypoints(sum);
  int i = 0;
  Mat voc;
  while (i < sum)
  {
    DetectKeypointsOnImage(filesList[i], keypoints[i], descriptors[i]);
    i++;
  }
  printf("DetectKeypointsOnImage\n");
  i = 0;
  voc = BuildVocabulary(descriptors, vocsize, tr1 + tr2 + tr3 + tr4);
  printf("BuildVocabulary\n");
  while (i < sum)
  {
    ComputeImgDescriptor(filesList[i], voc, samples.row(i));
    i++;
  }
  printf("ComputeImgDescriptor\n");
  Mat labels(sum, 1, CV_32S);
  int ind = 0;
  for (int j = 0 ; j < tr1; j++, ind++)
    labels.at<int>(ind) = 1;
  for (int j = 0; j < tr2; j++, ind++)
    labels.at<int>(ind) = 2;
  for (int j = 0; j < tr3; j++, ind++)
    labels.at<int>(ind) = 3;
  for (int j = 0; j < tr4; j++, ind++)
    labels.at<int>(ind) = 4;
  for (int j = 0; j < te1; j++, ind++)
    labels.at<int>(ind) = 1;
  for (int j = 0; j < te2; j++, ind++)
    labels.at<int>(ind) = 2;
  for (int j = 0; j < te3; j++, ind++)
    labels.at<int>(ind) = 3;
  for (int j = 0; j < te4; j++, ind++)
    labels.at<int>(ind) = 4;
  CvRTrees rf;
  Ptr<CvRTrees> rtp = &rf;
  printf("labels\n");
  TrainClassifier(samples, labels, trsum, tesum, vocsize, rtp);
  printf("TrainClassifier\n");
  rf.save("model-rf.yml", "simpleRTreesModel");
  printf("save\n");
  float trainError = 0.0f;
  for (int i = 0; i < trsum; ++i) {
    int prediction = (int)(rf.predict(samples.row(i))); 
    trainError += (labels.at<int>(i) != prediction); 
  } 
  trainError /= float(trsum);
  float testError = 0.0f; 
  for (int i = 0; i < sum - trsum; ++i) {
    int prediction = (int)(rf.predict(samples.row(trsum + i)));
    testError += (labels.at<int>(trsum + i) != prediction);
  } 
  testError /= float(sum - trsum);
  printf("train error = %.4f\ntest error = %.4f\n", trainError, testError);
  return 0;

}