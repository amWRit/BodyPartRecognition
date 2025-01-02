#include <iostream>
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include <opencv2/opencv.hpp>
#include "Image.h"
#include "Globals.h"
#include "Classifier.h"
#include <cstdlib> // For system()

int main()
{
    const std::string train_data_directory = "bodyparts-train-data"; // Fixed directory for training data
    const std::string test_data_directory = "bodyparts-test-data"; // Fixed directory for test data
    const std::string train_classification_file = "train_classication.csv"; // csv with classication for test data
    const std::string test_classification_file = "test_filenames.csv"; // csv with classication for test data
    const std::string images_dir = "images"; // Fixed directory for test images
    const std::string dicom_images_dir = "dicom-images"; // Fixed directory for test images

    std::vector<cv::Mat> images, trainImages, testImages;
    std::vector<int> labels, trainLabels, testLabels;

    // Load data from the specified directory
    build_data_class_ID_map(train_classification_file);
    load_data(train_data_directory, images, labels);

    // Split data into training and testing sets
    split_data(images, labels, trainImages, trainLabels, testImages, testLabels);

    // Check if any images were loaded
    if (trainImages.empty() || testImages.empty()) {
        std::cerr << "Error: No images loaded from directory!" << std::endl;
        return -1;
    }

    //Prepare PNG images from test DCM images
    convertDcmUsingCommandLine(dicom_images_dir);

    std::cout << "\n=== SVM MODEL ===\n";
    train_model(trainImages, trainLabels, ModelType::SVM);
    evaluate_model(testImages, testLabels, ModelType::SVM);
    make_predictions_on_loaded_set(images, labels, NUM_PREDICT_CASES, ModelType::SVM);
    make_predictions_on_test_cases(images_dir, ModelType::SVM);
    make_predictions_on_test_cases(dicom_images_dir + "/temp", ModelType::SVM);
    
    std::cout << "\n=== KNN MODEL ===\n";
    train_model(trainImages, trainLabels, ModelType::KNN);
    evaluate_model(testImages, testLabels, ModelType::KNN);
    make_predictions_on_loaded_set(images, labels, NUM_PREDICT_CASES, ModelType::KNN);
    make_predictions_on_test_cases(images_dir, ModelType::KNN);
    make_predictions_on_test_cases(dicom_images_dir + "/temp", ModelType::SVM);

}

