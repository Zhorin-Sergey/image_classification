# Image Classification Project

```bash
git clone https://github.com/Zhorin-Sergey/image_classification
mkdir image_classification_build
cd image_classification_build
cmake -DOpenCV_DIR=<opencv-dir> -G <generator> ..\image_classification
# <opencv-dir> - path to OpenCVConfig.cmake
# <generator> - one of the generators supported by CMake
#               to see all generators "cmake -G"
```