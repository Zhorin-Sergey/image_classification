#include "bow.h"

using namespace std;
using namespace cv;

// fileName – путь к изображению 
// keypoints – найденные ключевые точки на изображении
// img – исходное изображение
// descriptors – вычисленные значения дескрипторов ключевых точек 
void DetectKeypointsOnImage(const string& fileName, vector<KeyPoint>& keypoints, Mat& descriptors, char* DescriptorExtractorType, char* DetectorType)
{ 
  Mat img = imread(fileName);   // загружаем изображение из файла 
  initModule_nonfree();   // инициализируем модуль nonfree для использования детектора SIFT 
  Ptr<FeatureDetector> featureDetector = FeatureDetector::create(DetectorType);   // создаем SIFT детектор 
  featureDetector->detect(img, keypoints);   // детектируем ключевые точки на загруженном изображении 
  Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create(DescriptorExtractorType);  // создаем объект класса вычисления SIFT дескрипторов 
  descExtractor->compute(img, keypoints, descriptors);    // вычисляем дескрипторы ключевых точек на загруженном изображении 
}

// img – исходное изображение
// keypoints – ключевые точки на изображении
// descriptors – вычисленные значения дескрипторов ключевых точек 
void ComputeKeypointDescriptorsOnImage(const string& fileName, vector<KeyPoint>& keypoints, Mat& descriptors, char* DescriptorExtractorType)
{ 
  // инициализируем модуль nonfree для использования 
  // дескрипторов SIFT (если данная функция ранее не вызывалась) 
  initModule_nonfree();
  Mat img = imread(fileName);
  // создаем объект класса вычисления SIFT дескрипторов 
  Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create(DescriptorExtractorType);
  // вычисляем дескрипторы ключевых точек на загруженном изображении 
  descExtractor->compute(img, keypoints, descriptors);
}

// descriptors – предвычисленный набор дескрипторов изображений, используемый при построении словаря; 
// vocSize – размер словаря; 
// n - количество дискрипторов, используемых при обучении;
Mat BuildVocabulary(const std::vector<Mat>& descriptors, int vocSize, size_t n)
{ 
  
  BOWKMeansTrainer bowTrainer(vocSize); // создаем объект класса BOWTrainer с числом кластеров, равным vocSize, и остальными параметрами, используемыми по умолчанию 
  
  for (size_t i = 0; i < n; i++) { 
    bowTrainer.add(descriptors[i]); // добавляем дескрипторы особых точек с изображений, используемых при обучении словаря 
  } 
 
  Mat voc = bowTrainer.cluster();  // вызываем метод обучения словаря
  return voc; 
}

// img – исходное изображение 
// voc – словарь дескрипторов ключевых точек; 
// imgDesc – вычисленное признаковое описание изображения 
void ComputeImgDescriptor(const string& fileName, Mat& voc, Mat& imgDesc, char* DescriptorExtractorType, char* DetectorType)
{
  Mat img = imread(fileName);
  Ptr<FeatureDetector> featureDetector = FeatureDetector::create(DetectorType);   // создаем SIFT детектор ключевых точек 
  Ptr<DescriptorExtractor> dExtractor = DescriptorExtractor::create(DescriptorExtractorType); // создаем объект класса вычисления SIFT дескрипторов ключевых точек 
  Ptr<DescriptorMatcher> descriptorsMatcher = DescriptorMatcher::create("BruteForce");   // создаем объект класса, находящего ближайший к дескриптору центроид (по L2 метрике) 
  Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor( dExtractor, descriptorsMatcher);   // создаем объект класса, вычисляющего признаковое описание изображений
  bowExtractor->setVocabulary(voc);   // устанавливаем используемый словарь  дескрипторов ключевых точек 
  vector<KeyPoint> keypoints; 
  featureDetector->detect(img, keypoints);   // находим ключевые точки на изображении
  bowExtractor->compute(img, keypoints, imgDesc);   // вычисляем признаковое описание изображения
}
