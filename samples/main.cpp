#include <list>
#include <stdio.h>

#include <opencv2/opencv.hpp>

#include "utilities.h"
#include "bow.h"
#include "Classifier.h"

using namespace cv;

#define train1 "c:/Temp/Zhorin/data/train/indoor-dwelling-train"
#define train2 "c:/Temp/Zhorin/data/train/indoor-msu-train"
#define train3 "c:/Temp/Zhorin/data/train/outdoor-blossoms-train"
#define train4 "c:/Temp/Zhorin/data/train/outdoor-urban-train"
#define test1 "c:/Temp/Zhorin/data/test/indoor-dwelling-test" 
#define test2 "c:/Temp/Zhorin/data/test/indoor-msu-test" 
#define test3 "c:/Temp/Zhorin/data/test/outdoor-blossoms-test"
#define test4 "c:/Temp/Zhorin/data/test/outdoor-urban-test"
#define vocsize 200


int savecsv(Mat, char *);
int savecsvc(Mat, char *);

int main(int argc, char* argv[])
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
  
  
  std::cout << "START: DetectKeypointsOnImage (SURF)" << std::endl;
  std::vector<Mat> descriptors(trsum);
  std::vector<std::vector<KeyPoint>> keypoints(trsum);
  int i = 0;  
  Mat voc;
  while (i < trsum)
  {
    DetectKeypointsOnImage(filesList[i], keypoints[i], descriptors[i], "SURF", "SURF");    
    i++;
  }
  std::cout << "FINISH: DetectKeypointsOnImage (SURF)" << std::endl;


  std::cout << "START: BuildVocabulary" << std::endl;
  voc = BuildVocabulary(descriptors, vocsize, trsum);
  std::cout << "FINISH: BuildVocabulary" << std::endl;


  std::cout << "START: ComputeImgDescriptor for train" << std::endl;
  i = 0;
  Mat trainSamples;
  while (i < trsum)
  {
    Mat sample;
    ComputeImgDescriptor(filesList[i], voc, sample, "SURF", "SURF");
    trainSamples.push_back(sample);
    i++;
  }
  std::cout << "FINISH: ComputeImgDescriptor for train" << std::endl;
  savecsv(trainSamples, "train.csv");

  std::cout << "START: ComputeImgDescriptor for test" << std::endl;
  int j = 0;
  Mat testSamples;
  while (i < sum)
  {
    Mat sample;
    ComputeImgDescriptor(filesList[i], voc, sample, "SURF", "SURF");
    testSamples.push_back(sample);
    i++;
    j++;
  }
  std::cout << "FINISH: ComputeImgDescriptor for test" << std::endl;
  savecsv(testSamples, "test.csv");
  
    
  
  Mat trainLabels;
  Mat testLabels;
  int ind = 0;
  while (ind < tr1)
  {
    trainLabels.push_back(1);
    ind++;
  }
  while (ind < tr1 + tr2)
  {
    trainLabels.push_back(2);
    ind++;
  }
  while (ind < tr1 + tr2 + tr3)
  {
    trainLabels.push_back(3);
    ind++;
  }
  while (ind < tr1 + tr2 + tr3 + tr4)
  {
    trainLabels.push_back(4);
    ind++;
  }
  savecsvc(trainLabels, "trainlabels.csv");

  ind = 0;
  while (ind < te1)
  {
    testLabels.push_back(1);
    ind++;
  }
  while (ind < te1 + te2)
  {
    testLabels.push_back(2);
    ind++;
  }
  while (ind < te1 + te2 + te3)
  {
    testLabels.push_back(3);
    ind++;
  }
  while (ind < te1 + te2 + te3 + te4)
  {
    testLabels.push_back(4);
    ind++;
  }
  savecsvc(testLabels, "testlabels.csv");



  CvRTrees* rf = new CvRTrees();/*
  CvRTParams params = CvRTParams();
  params.term_crit.type = CV_TERMCRIT_ITER;
  params.term_crit.max_iter = 200;
  int numFeatures = trainSamples.cols;
  Mat varType(1, numFeatures + 1, CV_8UC1);
  for (int i = 0; i < numFeatures; i++)
  {
    // задаем тип признаков: вещественные
    varType.at<unsigned char>(i) = CV_VAR_ORDERED;
  }
  // задаем тип ответа: категориальный
  varType.at<unsigned char>(numFeatures) = CV_VAR_CATEGORICAL;
  std::cout << "START: Train RF" << std::endl;
  rf->train(trainSamples, CV_ROW_SAMPLE, trainLabels, Mat(), Mat(), varType, Mat(), params);
  std::cout << "FINISH: Train RF" << std::endl;

  rf->save("model-rf.yml", "simpleRTreesModel");
 */
    rf->load("model-rf.yml", "simpleRTreesModel");
  float trainError = 0.0f;
  for (int i = 0; i < trsum; ++i) {
    int prediction = (int)(rf->predict(trainSamples.row(i)));
    printf("%i ", prediction);
    trainError += (trainLabels.at<int>(i) != prediction);
  }
  printf("TEST\n");
  trainError /= float(trsum);
  float testError = 0.0f; 
  for(int i = 0; i <tesum; ++i) {
    int prediction = (int)(rf->predict(testSamples.row(i)));
    printf("%i ", prediction);
    testError += (testLabels.at<int>(i) != prediction);
  }
  testError /= float(sum - trsum);
  printf("train error = %.4f\ntest error = %.4f\n", trainError, testError);
  return 0;
}

int savecsv(Mat samples, char *filename)
{
  std::cout << "START: savecsv " << filename << std::endl;
  FILE *f = fopen(filename, "w");
  for (int i = 0; i < samples.rows; i++)
  {
    for (int j = 0; j < samples.cols; j++)
    {
      fprintf(f, "%f;", samples.at<float>(i, j));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  std::cout << "FINISH: savecsv " << filename << std::endl;
  return 0;
}

int savecsvc(Mat samples, char *filename)
{
  std::cout << "START: savecsvc " << filename << std::endl;
  FILE *f = fopen(filename, "w");
  for (int i = 0; i < samples.rows; i++)
  {
    for (int j = 0; j < samples.cols; j++)
    {
      fprintf(f, "%d;", samples.at<int>(i));
    }
    fprintf(f, "\n");
  }
  fclose(f);
  std::cout << "FINISH: savecsvc " << filename << std::endl;
  return 0;
}