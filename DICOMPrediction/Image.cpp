#include "dcmtk/dcmdata/dctk.h" // Include DCMTK headers
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <filesystem>
#include <unordered_map>
#include <random>
#include <opencv2/ml.hpp>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>
#include <chrono>
#include "EnumClass.h"
#include "Image.h"
#include "Classifier.h"
#include "Globals.h"

#include <cstdlib> // For system()

namespace fs = std::filesystem;

// Function to build a map containing test data image name and its classID from train_classification.csv
void build_data_class_ID_map(const std::string& filePath) {
    std::cout << "\nBuilding Test Data Classication Map...\n";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    std::string line;
    // Skip header
    std::getline(file, line);
    int count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string fileName, classID_str;

        if (std::getline(ss, fileName, ',') && std::getline(ss, classID_str, ','))
        {
            // Create a stringstream for the classID string to parse multiple IDs
            std::stringstream classID_stream(classID_str);
            std::string id;
            std::vector<int> ids;

            // Parse multiple IDs separated by some delimiter (assuming space)
            // Modify the delimiter based on your data format
            while (std::getline(classID_stream, id, ' ')) {
                try {
                    ids.push_back(std::stoi(id));
                }
                catch (const std::exception& e) {
                    std::cerr << "Error converting ID: " << id << " for file: " << fileName << std::endl;
                    continue;
                }
            }

            classID_map[fileName] = ids;
            count++;
        }
    }
    std::cout << "Building map completed.\n";
    std::cout << "Size: " << classID_map.size() << std::endl;
    std::cout << "Example:\n";

    count = 0;
    for (const auto& entry : classID_map) {
        std::cout << entry.first << " : ";
        for (size_t i = 0; i < entry.second.size(); ++i) {
            std::cout << entry.second[i];
            if (i < entry.second.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "\n";
        count++;
        if (count >= 5) break;
    }
    file.close();
}

// Function to load the PNG images of body parts inside bodyparts-train-data directory
void load_data(const std::string& data_dir,
    std::vector<cv::Mat>& images,
    std::vector<int>& labels) {

    std::cout << "\nLoading training images and labels...\n";

    // Convert to absolute path
    fs::path dir_path = fs::absolute(fs::path(data_dir));
    std::cout << "\nDirectory details:" << std::endl;
    std::cout << "Input path: " << data_dir << std::endl;
    std::cout << "Absolute path: " << dir_path.string() << std::endl;
    std::cout << "Current working directory: " << fs::current_path().string() << std::endl;

    if (!fs::exists(dir_path)) {
        std::cerr << "Error: Directory does not exist: " << dir_path.string() << std::endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".png") {
            fs::path fullPath = entry.path();
            std::string fileName = entry.path().filename().string();
            cv::Mat img;
            img = cv::imread(fullPath.string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Could not open or find image: " << entry.path() << std::endl;
                continue; // Skip this image if it cannot be loaded;
            }

            // Check if the image has an entry in the classID map
            auto mapIt = classID_map.find(fileName);
            if (mapIt == classID_map.end()) {
                std::cerr << "Warning: No class ID found for image: " << fileName << std::endl;
                continue;
            }

            // For each label associated with this image, create a training sample
            for (const int& label : mapIt->second) {
                preprocess_image(img);
                images.push_back(img.clone());  // Need to clone for separate instances
                labels.push_back(label);
            }
        }
    }

    std::cout << "Loading completed.\n";
    std::cout << "Data Rows: " << images.size() << ", Labels Rows: " << labels.size() << std::endl;
}


//// Preprocess image (resize and normalize if necessary)
void preprocess_image(cv::Mat& img) {
    // Convert to grayscale if it's a color image
    if (img.channels() == 3) { // Check if the image has 3 channels (BGR)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // Convert to grayscale
    }
    else if (img.channels() != 1) { // If not grayscale and not empty, handle unexpected cases
        std::cerr << "Warning: Unexpected number of channels in image: " << img.channels() << std::endl;
        return; // Skip processing if channels are unexpected
    }

    cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    img.convertTo(img, CV_32F); // Convert to float
    img /= 255.0;

    // Apply histogram equalization
    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

    // Add Gaussian blur to reduce noise
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0);

    // Enhance edges using Sobel
    cv::Mat gradX, gradY;
    cv::Sobel(img, gradX, CV_32F, 1, 0);
    cv::Sobel(img, gradY, CV_32F, 0, 1);
    cv::addWeighted(img, 0.7, (gradX + gradY), 0.3, 0, img);

    // Ensure values are still in [0,1] range
    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

    // print_data_statistics(img, "After preprocessing");

    // Flatten the image
    img = img.reshape(1, 1);

}

// Function to split data into training and testing sets
void split_data(const std::vector<cv::Mat>& images, const std::vector<int>& labels,
    std::vector<cv::Mat>& trainImages, std::vector<int>& trainLabels,
    std::vector<cv::Mat>& testImages, std::vector<int>& testLabels) {
    std::cout << "\nSplitting into Train and Test images and labels...\n";
    // Create a vector of indices
    std::vector<int> indices(images.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Calculate the split index
    size_t splitIndex = static_cast<size_t>(images.size() * (1 - TEST_SIZE));

    // Split into training and testing sets
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < splitIndex) {
            trainImages.push_back(images[indices[i]]);
            trainLabels.push_back(labels[indices[i]]);
        }
        else {
            testImages.push_back(images[indices[i]]);
            testLabels.push_back(labels[indices[i]]);
        }
    }
    std::cout << "Splitting completed.\n";
    std::cout << "Train Data Rows: " << trainImages.size() << ", Train Labels Rows: " << trainLabels.size() << std::endl;
    std::cout << "Test Data Rows: " << testImages.size() << ", Test Labels Rows: " << testLabels.size() << std::endl;

}

// Function to convert DICOM data to cv::Mat
cv::Mat convertDcmToMat(DcmDataset* dataset) {
    DcmElement* pixelData = nullptr; // Declare a pointer for pixel data
    OFCondition status = dataset->findAndGetElement(DCM_PixelData, pixelData); // Get pixel data

    // Check if the status is normal and pixelData is not null
    if (status.bad() || pixelData == nullptr) {
        std::cerr << "Error: Pixel data not found." << std::endl;
        return cv::Mat(); // Return an empty Mat if pixel data is not found
    }

    // Get image dimensions
    Uint16 rows, cols;
    dataset->findAndGetUint16(DCM_Rows, rows);
    dataset->findAndGetUint16(DCM_Columns, cols);

    // Create a Mat object for grayscale image
    cv::Mat img(rows, cols, CV_8UC1); // Assuming 8-bit grayscale

    // Copy pixel data into the Mat object
    //pixelData->getUint8Array(reinterpret_cast<Uint8*>(img.data));

    // Copy pixel data into the Mat object
    if (pixelData) {
        std::memcpy(img.data, pixelData, rows * cols); // Copy pixel data to OpenCV Mat
    }
    else {
        std::cerr << "Error: Pixel data is null." << std::endl;
        return cv::Mat(); // Return an empty Mat if pixel data is null
    }

    return img;
}

void convertDcmUsingCommandLine(const std::string& dicom_dir) {
    std::cout << "\n Converting DCM images to PNG...\n";

    // Create a temporary folder inside dicom_dir
    std::string tempDir = dicom_dir + "/temp";
    if (std::filesystem::exists(tempDir))
        std::filesystem::remove_all(tempDir);
    std::filesystem::create_directory(tempDir);       

    for (const auto& entry : std::filesystem::directory_iterator(dicom_dir)) {
        if (entry.path().extension() == ".dcm") { // Only load .dcm files
            const std::string filePath = entry.path().string();
            DcmFileFormat fileformat;

            if (fileformat.loadFile(filePath.c_str()).bad()) {
                std::cerr << "Error: Cannot read DICOM file: " << filePath << std::endl;
                continue; // Skip this file if it cannot be loaded
            }

            std::string outputImageType = "png";
            const std::string dicomFilePath = entry.path().string();
            const std::string outPutFilePath = tempDir + "/" + entry.path().filename().stem().string() + "." + outputImageType;

            std::string command = "dcm2img +on2 +Wm +a +f8 " + dicomFilePath + " " + outPutFilePath;
            int result = system(command.c_str());
            //std::cout << command << std::endl;
            if (result == 0) {
                //std::cout << "Successfully converted " << dicomFilePath << " to " << outPutFilePath << std::endl;
            }
            else {
                std::cerr << "Error converting file using dcm2img." << std::endl;
            }

        }
    }

    std::cout << "Conversion completed.\n";
}

void convertDCM2PNG(const std::string& dicomPath, const std::string& outputPath) {
    // https://support.dcmtk.org/docs/dcm2img.html
    //std::string command = "dcm2img +on2 +Wm +a +f8 " + dicomPath + " " + outputPath;
    std::string command = "dcm2img +on2 +Wm +a +f8 \"" + dicomPath + "\" \"" + outputPath + "\"";
    int result = system(command.c_str());
    std::cout << command << std::endl;
    if (result == 0) {
        std::cout << "Successfully converted " << dicomPath << " to " << outputPath << std::endl;
    }
    else {
        std::cerr << "Error converting file using dcm2img." << std::endl;
    }
}