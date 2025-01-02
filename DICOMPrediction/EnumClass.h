#pragma once
#include <iostream>
#include <memory>
#include <unordered_map>

//enums for ModelType
enum class ModelType {
    KNN,
    SVM,
    Unknown
};

const std::unordered_map<ModelType, std::string> modelTypeToString = {
    {ModelType::KNN, "KNN"},
    {ModelType::SVM, "SVM"},
};

inline std::string toString(ModelType modelType) {
    auto it = modelTypeToString.find(modelType);
    return (it != modelTypeToString.end()) ? it->second : "Unknown";
}
