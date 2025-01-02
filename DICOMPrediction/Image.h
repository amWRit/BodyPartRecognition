#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include "EnumClass.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"

void load_data(const std::string& data_dir,
	std::vector<cv::Mat>& images,
	std::vector<int>& labels);

void split_data(const std::vector<cv::Mat>& images, const std::vector<int>& labels,
	std::vector<cv::Mat>& trainImages, std::vector<int>& trainLabels,
	std::vector<cv::Mat>& testImages, std::vector<int>& testLabels);

void convertDcmUsingCommandLine(const std::string& dicom_dir);

cv::Mat convertDcmToMat(DcmDataset* dataset);

void build_data_class_ID_map(const std::string& filePath);

void split_data(const std::vector<cv::Mat>& images, const std::vector<int>& labels,
	std::vector<cv::Mat>& trainImages, std::vector<int>& trainLabels,
	std::vector<cv::Mat>& testImages, std::vector<int>& testLabels);

void preprocess_image(cv::Mat& img);


#endif // IMAGE_H