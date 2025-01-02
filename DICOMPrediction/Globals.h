#pragma once
#ifndef GLOBALS_H
#define GLOBALS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <opencv2/ml.hpp> // Include OpenCV ML headers

// Declare external variables
extern std::unordered_map<std::string, std::vector<int>> classID_map;
extern cv::Ptr<cv::ml::KNearest> knn;
extern cv::Ptr<cv::ml::SVM> svm;

const int IMG_WIDTH = 30;
const int IMG_HEIGHT = 30;
const int NUM_CATEGORIES = 43;
const float TEST_SIZE = 0.1f; // 20% of the data will be used for testing
const int NUM_TEST_CASES = 1000;
const int NUM_PREDICT_CASES = 5;

#endif // GLOBALS_H
