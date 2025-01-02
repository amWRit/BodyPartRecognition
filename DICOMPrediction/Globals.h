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

// Map for body parts
const std::unordered_map<int, std::string> BODY_PARTS = {
    {0, "Abdomen"},
    {1, "Ankle"},
    {2, "Cervical Spine"},
    {3, "Chest"},
    {4, "Clavicles"},
    {5, "Elbow"},
    {6, "Feet"},
    {7, "Finger"},
    {8, "Forearm"},
    {9, "Hand"},
    {10, "Hip"},
    {11, "Knee"},
    {12, "Lower Leg"},
    {13, "Lumbar Spine"},
    {14, "Others"},
    {15, "Pelvis"},
    {16, "Shoulder"},
    {17, "Sinus"},
    {18, "Skull"},
    {19, "Thigh"},
    {20, "Thoracic Spine"},
    {21, "Wrist"}
};

#endif // GLOBALS_H
