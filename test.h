#ifndef TEST_H_
#define TEST_H_

#include "opencv.hpp"
#include "cv.h"
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include <iostream>
#include <io.h>

#define CELL_SIZE  8
#define BLOCK_SIZE 16
#define BIN_NUM 9
#define ANGLE_SCALE (180/BIN_NUM)

using namespace std;
using namespace cv;

void ComputeHogFeatures(Mat &roi,vector<float> &features);
bool get_filelist_from_dir(string path, vector<string>& files);

#endif
