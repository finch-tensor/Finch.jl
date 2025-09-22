#include "bench.h"

#include <cstdlib>
#include <iostream>

#include "taco/tensor.h"
#include "taco/util/strings.h"

std::string getEnvVar(std::string varname) {
    auto path = std::getenv(varname.c_str());
    if (path == nullptr) {
        return "";
    }
    return std::string(path);
}

std::string getSSTensorPATH() {
    std::string result = getEnvVar("SUITESPARSE_PATH");
    if (result == "") {
        assert(false && "SUITESPARSE_PATH is unset");
    }
    return cleanPath(result);
}

std::string getFrosttTensorPATH() {
    std::string result = getEnvVar("FROSTT_PATH");
    if (result == "") {
        assert(false && "FROSTT_PATH is unset");
    }
    return cleanPath(result);
}

std::string getSSOtherTensorPATH() {
    std::string result = getEnvVar("SUITESPARSE_OTHER_PATH");
    if (result == "") {
        assert(false && "FROSTT_PATH is unset");
    }
    return cleanPath(result);
}

std::string getFrosttOtherTensorPATH() {
    std::string result = getEnvVar("FROSTT_OTHER_PATH");
    if (result == "") {
        assert(false && "FROSTT_PATH is unset");
    }
    return cleanPath(result);
}

std::string getTacoTensorPath() {
    std::string result = getEnvVar("TACO_TENSOR_PATH");
    if (result == "") {
        assert(false && "TACO_TENSOR_PATH is unset");
    }
    return cleanPath(result);
}

std::string getValidationOutputPath() {
    auto result = getEnvVar("VALIDATION_OUTPUT_PATH");
    if (result != "") {
        result = cleanPath(result);
    }
    return result;
}

std::string cleanPath(std::string path) {
    std::string result(path);
    if (result[result.size() - 1] != '/') {
        result += "/";
    }
    return result;
}


std::string constructRandomTensorKey(std::vector<int> dims, float sparsity, int variant) {
    auto path = getTacoTensorPath();
    std::stringstream result;
    result << path;
    if (path[path.size() - 1] != '/') {
        result << "/";
    }
    result << "random/";
    if (variant == 0) {
        result << taco::util::join(dims, "x") << "-" << sparsity << ".tns";
    } else {
        result << taco::util::join(dims, "x") << "-" << sparsity << "-" << variant << ".tns";
    }
    return result.str();
}

taco::TensorBase loadRandomTensor(std::string name, std::vector<int> dims, float sparsity, taco::Format format, int variant) {
    // For now, just say that the python code must generate the random
    // tensor before use.
    auto tensor = taco::read(constructRandomTensorKey(dims, sparsity, variant), format, true);
    tensor.setName(name);
    return tensor;
}

std::string constructImageTensorKey(int num, int variant, float threshold) {
    auto path = getTacoTensorPath();
    std::stringstream result;
    result << path;
    if (path[path.size() - 1] != '/') {
        result << "/";
    }
    result << "image/tensors/";
    if (variant == 0) {
        result << "image" << num  << "-" << threshold << ".tns";
    } else if (variant == 3) {
        result << "image" << num << "-" << variant << ".tns";
    } else {
        result << "image" << num << "-" << threshold << "-" << variant << ".tns";
    }
    return result.str();
}

taco::TensorBase loadImageTensor(std::string name, int num, taco::Format format, float threshold, int variant) {
    // For now, just say that the python code must generate the random
    // tensor before use.
    auto tensor = taco::read(constructImageTensorKey(num, variant, threshold), format, true);
    tensor.setName(name);
    return tensor;
}

std::string constructMinMaxTensorKey(int order, int variant) {
    auto path = getTacoTensorPath();
    std::stringstream result;
    result << path;
    if (path[path.size() - 1] != '/') {
        result << "/";
    }
    result << "minmax/";
    if (variant == 0) {
        result << "minmax-"  << order << ".tns";
    } else {
        result << "minmax-" << order << "-" << variant << ".tns";
    }
    return result.str();
}

taco::TensorBase loadMinMaxTensor(std::string name, int order, taco::Format format, int variant) {
    // For now, just say that the python code must generate the random
    // tensor before use.
    auto tensor = taco::read(constructMinMaxTensorKey(order, variant), format, true);
    tensor.setName(name);
    return tensor;
}

std::string constructOtherVecKey(std::string tensorName, std::string variant, float sparsity) {
    auto path = getTacoTensorPath();
    std::stringstream result;
    result << path;
    if (path[path.size() - 1] != '/') {
        result << "/";
    }
    result << "other/";
    result << tensorName << "-" << variant << "-" << sparsity << ".tns";
    return result.str();
}

std::string constructOtherMatKey(std::string tensorName, std::string variant, std::vector<int> dims, float sparsity) {
    auto path = getTacoTensorPath();
    std::stringstream result;
    result << path;
    if (path[path.size() - 1] != '/') {
        result << "/";
    }
    result << "other/";
    result << tensorName << "-" << variant << "-" << taco::util::join(dims, "x") << "-" << sparsity << ".tns";
    return result.str();
}
