#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include "EnumClass.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "Globals.h"


void configure_svm();
void configure_knn();

void train_model(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels,
	ModelType modelType);

void evaluate_model(const std::vector<cv::Mat>& testImages,
	const std::vector<int>& testLabels,
	ModelType modelType);

int predict_body_part(const cv::Mat& img, ModelType modelType);

void make_predictions_on_loaded_set(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels, int count, ModelType modelType);

void make_predictions_on_test_cases(const std::string& images_dir, ModelType modelType);

std::pair<double, double> optimize_svm_parameters(const cv::Mat& trainData,
	const cv::Mat& labelsMat,
	double& bestAccuracy);

#endif // CLASSIFIER_H