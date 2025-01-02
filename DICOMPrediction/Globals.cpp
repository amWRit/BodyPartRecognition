#include "Globals.h"

// Define the global variables
std::unordered_map<std::string, std::vector<int>> classID_map; // Definition of classID_map
cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create(); // Initialize KNN
cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create(); // Initialize SVM