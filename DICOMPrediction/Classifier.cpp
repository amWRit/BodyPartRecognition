#include "Classifier.h"
#include "Image.h"
#include "Globals.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <unordered_map>

// function for training the model
void train_model(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels,
    ModelType modelType) {
    // Prepare data for training
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "\nTraining the model...\n";
    cv::Mat trainData;
    cv::Mat labelsMat = cv::Mat(labels).reshape(1, labels.size()); // Convert labels to Mat

    // Flatten images into a single row
    for (const auto& img : images) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten to 1D
        trainData.push_back(flatImg);
    }

    // print_data_statistics(trainData, "Training data");

    // Normalize pixel values
    //trainData /= 255.0; // Normalize pixel values to [0, 1]

    // Convert to float if necessary
    if (trainData.type() != CV_32F) {
        trainData.convertTo(trainData, CV_32F);
    }

    // Check if trainData and labelsMat are properly formed
    if (trainData.empty() || labelsMat.empty()) {
        std::cerr << "Training data or labels are empty!" << std::endl;
        return;
    }

    std::cout << "Train Data Rows: " << trainData.rows << ", Train Labels Rows: " << labelsMat.rows << std::endl;

    // Create and configure classifiers and train the model
    if (modelType == ModelType::SVM)
    {
        configure_svm();
        svm->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);
    }
    else if (modelType == ModelType::KNN) {
        configure_knn();
        knn->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() / 60000000 << " minutes" << std::endl;
}

// Function for evaluating the trained model by testing against splitted test data
void evaluate_model(const std::vector<cv::Mat>& testImages,
    const std::vector<int>& testLabels,
    ModelType modelType) {
    // Prepare data for evaluation
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "\nEvaluating the model (splitted train vs test data)...\n";
    cv::Mat testData;

    // Flatten test images into a single row
    for (const auto& img : testImages) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten to 1D
        testData.push_back(flatImg);
    }

    // Normalize pixel values
    //testData /= 255.0; // Normalize pixel values to [0, 1]

    // Convert to float if necessary
    if (testData.type() != CV_32F) {
        testData.convertTo(testData, CV_32F);
    }

    // Check if testData is empty
    if (testData.empty()) {
        std::cerr << "Test data is empty!" << std::endl;
        return;
    }

    // == WITHOUT THREADING ==
    // Predict labels for the test data using multiple threads
    //cv::Mat predictedLabels;
    //knn->findNearest(testData, knn->getDefaultK(), predictedLabels);

    //// Calculate accuracy
    //int correctCount = 0;
    //for (size_t i = 0; i < predictedLabels.rows; ++i) {
    //    // std::cout << predictedLabels.at<float>(i, 0) << " : " << testLabels[i] << "\n";
    //    if (predictedLabels.at<float>(i, 0) == testLabels[i]) {
    //        correctCount++;
    //    }
    //}
    //// == WITHOUT THREADING ==

    // == WITH THREADING ==
    // Number of threads to use
    const int numThreads = std::thread::hardware_concurrency(); // Get number of hardware threads
    const size_t totalImages = testImages.size();

    // Create a vector to hold futures for thread results
    std::vector<std::future<void>> futures(numThreads);
    int correctCount = 0;

    // Lambda function for processing images in parallel
    auto process_images = [&](size_t start, size_t end, int threadIndex) {
        std::cout << "Starting thread..." << threadIndex << "\n";
        cv::Mat localPredictedLabels;
        if (modelType == ModelType::SVM)
        {
            svm->predict(testData.rowRange(start, end), localPredictedLabels);
        }
        else if (modelType == ModelType::KNN) {
            knn->findNearest(testData.rowRange(start, end), knn->getDefaultK(), localPredictedLabels);
        }

        // Store results in a shared vector or process them directly here
        for (size_t i = start; i < end; ++i) {
            if (localPredictedLabels.at<float>(i - start, 0) == testLabels[i]) {
                // Increment correct count or store results as needed
                correctCount++;
            }
        }
        };

    // Divide work among threads
    size_t chunkSize = totalImages / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numThreads - 1) ? totalImages : start + chunkSize; // Handle last chunk

        futures[i] = std::async(std::launch::async, process_images, start, end, i);
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
    // == WITH THREADING ==

    double accuracy = static_cast<double>(correctCount) / testLabels.size() * 100.0;
    // Display results
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() / 60000000 << " minutes" << std::endl;
}


// Function to predict the body part category
int predict_body_part(const cv::Mat& img, ModelType modelType) {
    cv::Mat processedImg = img.clone(); // Clone to avoid modifying 

    // Ensure proper type and shape
    if (processedImg.type() != CV_32F) {
        processedImg.convertTo(processedImg, CV_32F);
    }
    // print_data_statistics(processedImg, "Training data");

    // Make prediction
    cv::Mat predictedLabel;

    if (modelType == ModelType::SVM)
    {
        svm->predict(processedImg, predictedLabel);
    }
    else if (modelType == ModelType::KNN) {
        knn->findNearest(processedImg, knn->getDefaultK(), predictedLabel);
    }

    return static_cast<int>(predictedLabel.at<float>(0, 0)); // Return predicted label
}

// Functions to make predictions for <Count> number of images loaded in the beginning
void make_predictions_on_loaded_set(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels, int count,
    ModelType modelType) {
    std::cout << "\nMaking predictions on random samples from loaded set...\n";
    // Create a vector of indices
    std::vector<int> indices(images.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (int i = 0; i < count; ++i) {

        cv::Mat img = images.at(indices[i]);
        if (img.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
        }

        // Predict the body part category
        int predicted_label = predict_body_part(img, ModelType::SVM);

        if (predicted_label == -1) {
            std::cerr << "Error: Prediction failed!" << std::endl;
        }

        std::cout << "Predicted Body Part ID: " << predicted_label << " || Actual label: " << labels[indices[i]] << std::endl;
    }
    std::cout << "Predictions completed.\n";
}

// Functions to make predictions for random images from /images folder
void make_predictions_on_test_cases(const std::string& images_dir, ModelType modelType) {
    //std::cout << "\nMaking predictions on test cases...\n";
    if (images_dir.find("temp") != std::string::npos) {
        std::cout << "\nMaking predictions on converted DCM test cases...\n";
    }
    else
        std::cout << "\nMaking predictions on PNG test cases...\n";

    for (const auto& entry : std::filesystem::directory_iterator(images_dir)) {
        if (entry.path().extension() == ".png") { // Only load .ppm files
            std::string filePath = entry.path().string();
            cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Could not open or find image: " << entry.path() << std::endl;
                continue; // Skip this image if it cannot be loaded
            }
            preprocess_image(img); // Preprocess image if needed
            // Predict the body part category
            int predicted_label = predict_body_part(img, ModelType::SVM);

            if (predicted_label == -1) {
                std::cerr << "Error: Prediction failed!" << std::endl;
            }
            std::cout << filePath << " || Predicted Body Part ID: " << predicted_label << std::endl;
        }
    }
    std::cout << "Predictions completed.\n";
}

// function for parameter optimization
std::pair<double, double> optimize_svm_parameters(const cv::Mat& trainData,
    const cv::Mat& labelsMat,
    double& bestAccuracy) {

    std::cout << "\nOptimizing SVM parameters using k-fold cross validation...\n";
    auto start = std::chrono::high_resolution_clock::now();
    // Define parameter grid
    std::vector<double> gammaValues = { 0.001, 0.01, 0.1, 1.0, 10.0 };
    std::vector<double> CValues = { 0.1, 1.0, 10.0, 100.0, 1000.0 };

    double bestC = 1.0;
    double bestGamma = 0.1;
    bestAccuracy = 0;

    // Number of folds for cross-validation
    const int k_folds = 5;

    // Calculate fold size
    int fold_size = trainData.rows / k_folds;

    // Grid search with k-fold cross-validation
    for (double C : CValues) {
        for (double gamma : gammaValues) {
            double total_accuracy = 0.0;

            std::cout << "Testing C=" << C << ", gamma=" << gamma << std::endl;

            // Perform k-fold cross-validation
            for (int fold = 0; fold < k_folds; fold++) {
                // Calculate validation set range
                int validation_start = fold * fold_size;
                int validation_end = (fold == k_folds - 1) ? trainData.rows : (fold + 1) * fold_size;

                // Create training and validation sets
                cv::Mat fold_train_data, fold_train_labels;
                cv::Mat fold_val_data, fold_val_labels;

                // Split data into training and validation
                for (int i = 0; i < trainData.rows; i++) {
                    if (i >= validation_start && i < validation_end) {
                        fold_val_data.push_back(trainData.row(i));
                        fold_val_labels.push_back(labelsMat.row(i));
                    }
                    else {
                        fold_train_data.push_back(trainData.row(i));
                        fold_train_labels.push_back(labelsMat.row(i));
                    }
                }

                // Configure and train SVM for this fold
                cv::Ptr<cv::ml::SVM> fold_svm = cv::ml::SVM::create();
                fold_svm->setType(cv::ml::SVM::C_SVC);
                fold_svm->setKernel(cv::ml::SVM::RBF);
                fold_svm->setC(C);
                fold_svm->setGamma(gamma);

                // Train on fold training data
                fold_svm->train(fold_train_data, cv::ml::ROW_SAMPLE, fold_train_labels);

                // Validate on fold validation data
                cv::Mat predictions;
                fold_svm->predict(fold_val_data, predictions);

                // Calculate accuracy for this fold
                int correct = 0;
                for (int i = 0; i < predictions.rows; i++) {
                    if (predictions.at<float>(i) == fold_val_labels.at<int>(i)) {
                        correct++;
                    }
                }
                double fold_accuracy = static_cast<double>(correct) / predictions.rows;
                total_accuracy += fold_accuracy;
            }

            // Calculate average accuracy across all folds
            double avg_accuracy = total_accuracy / k_folds;
            std::cout << "Average accuracy: " << avg_accuracy * 100 << "%" << std::endl;

            // Update best parameters if necessary
            if (avg_accuracy > bestAccuracy) {
                bestAccuracy = avg_accuracy;
                bestC = C;
                bestGamma = gamma;
            }
        }
    }

    std::cout << "Best parameters found: C=" << bestC
        << ", gamma=" << bestGamma
        << ", accuracy=" << bestAccuracy * 100 << "%" << std::endl;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() / 60000000 << " minutes" << std::endl;

    return std::make_pair(bestC, bestGamma);
}

// configure svm instance
void configure_svm() {
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);

    // Find optimal parameters
    double bestAccuracy;
    double bestC = 100, bestGamma = 0.01;
    //auto [bestC, bestGamma] = optimize_svm_parameters(trainData, labelsMat, bestAccuracy);

    // Train final model with best parameters
    svm->setC(bestC);
    svm->setGamma(bestGamma);
}

// configure knn instance
void configure_knn() {
    knn = cv::ml::KNearest::create();
    knn->setDefaultK(3); // Set number of neighbors
   // Train k-NN model with different k values
   //for (int k = 1; k <= 11; k += 2) { // Test odd values from 1 to 11
   //    knn->setDefaultK(k);
   //    knn->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);
   //}

    // Save the trained model if needed
   /*try {
       knn->save("knn_model.xml");
   }
   catch (const cv::Exception& e) {
       std::cerr << "Error saving model: " << e.what() << std::endl;
   }*/
}

