#include "bow.h"

using namespace std;
using namespace cv;

// fileName � ���� � ����������� 
// keypoints � ��������� �������� ����� �� �����������
// img � �������� �����������
// descriptors � ����������� �������� ������������ �������� ����� 
void DetectKeypointsOnImage(const string& fileName, vector<KeyPoint>& keypoints, Mat& descriptors, char* DescriptorExtractorType, char* DetectorType)
{ 
  Mat img = imread(fileName);   // ��������� ����������� �� ����� 
  initModule_nonfree();   // �������������� ������ nonfree ��� ������������� ��������� SIFT 
  Ptr<FeatureDetector> featureDetector = FeatureDetector::create(DetectorType);   // ������� SIFT �������� 
  featureDetector->detect(img, keypoints);   // ����������� �������� ����� �� ����������� ����������� 
  Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create(DescriptorExtractorType);  // ������� ������ ������ ���������� SIFT ������������ 
  descExtractor->compute(img, keypoints, descriptors);    // ��������� ����������� �������� ����� �� ����������� ����������� 
}

// img � �������� �����������
// keypoints � �������� ����� �� �����������
// descriptors � ����������� �������� ������������ �������� ����� 
void ComputeKeypointDescriptorsOnImage(const string& fileName, vector<KeyPoint>& keypoints, Mat& descriptors, char* DescriptorExtractorType)
{ 
  // �������������� ������ nonfree ��� ������������� 
  // ������������ SIFT (���� ������ ������� ����� �� ����������) 
  initModule_nonfree();
  Mat img = imread(fileName);
  // ������� ������ ������ ���������� SIFT ������������ 
  Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create(DescriptorExtractorType);
  // ��������� ����������� �������� ����� �� ����������� ����������� 
  descExtractor->compute(img, keypoints, descriptors);
}

// descriptors � ��������������� ����� ������������ �����������, ������������ ��� ���������� �������; 
// vocSize � ������ �������; 
// n - ���������� ������������, ������������ ��� ��������;
Mat BuildVocabulary(const std::vector<Mat>& descriptors, int vocSize, size_t n)
{ 
  
  BOWKMeansTrainer bowTrainer(vocSize); // ������� ������ ������ BOWTrainer � ������ ���������, ������ vocSize, � ���������� �����������, ������������� �� ��������� 
  
  for (size_t i = 0; i < n; i++) { 
    bowTrainer.add(descriptors[i]); // ��������� ����������� ������ ����� � �����������, ������������ ��� �������� ������� 
  } 
 
  Mat voc = bowTrainer.cluster();  // �������� ����� �������� �������
  return voc; 
}

// img � �������� ����������� 
// voc � ������� ������������ �������� �����; 
// imgDesc � ����������� ����������� �������� ����������� 
void ComputeImgDescriptor(const string& fileName, Mat& voc, Mat& imgDesc, char* DescriptorExtractorType, char* DetectorType)
{
  Mat img = imread(fileName);
  Ptr<FeatureDetector> featureDetector = FeatureDetector::create(DetectorType);   // ������� SIFT �������� �������� ����� 
  Ptr<DescriptorExtractor> dExtractor = DescriptorExtractor::create(DescriptorExtractorType); // ������� ������ ������ ���������� SIFT ������������ �������� ����� 
  Ptr<DescriptorMatcher> descriptorsMatcher = DescriptorMatcher::create("BruteForce");   // ������� ������ ������, ���������� ��������� � ����������� �������� (�� L2 �������) 
  Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor( dExtractor, descriptorsMatcher);   // ������� ������ ������, ������������ ����������� �������� �����������
  bowExtractor->setVocabulary(voc);   // ������������� ������������ �������  ������������ �������� ����� 
  vector<KeyPoint> keypoints; 
  featureDetector->detect(img, keypoints);   // ������� �������� ����� �� �����������
  bowExtractor->compute(img, keypoints, imgDesc);   // ��������� ����������� �������� �����������
}
