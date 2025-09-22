#include "test.h"
#include "taco/tensor.h"
#include "taco/codegen/module.h"
#include <taco/index_notation/transformations.h>
#include <fstream>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
#include "taco/lower/lower.h"
#include "op_factory.h"
#include <experimental/filesystem>

#include <tuple>
#include <ios>
using namespace taco;

template<typename T>
taco::Tensor<T> castToType(std::string name, taco::Tensor<double> tensor) {
    taco::Tensor<T> result(name, tensor.getDimensions(), tensor.getFormat());
    std::vector<int> coords(tensor.getOrder());
    for (auto& value : taco::iterate<double>(tensor)) {
        for (int i = 0; i < tensor.getOrder(); i++) {
            coords[i] = value.first[i];
        }
        // Attempt to cast the value to an integer. However, if the cast causes
        // the value to equal 0, then this will ruin the sparsity pattern of the
        // tensor, as the 0 values will get compressed out. So, if a cast would
        // equal 0, insert 1 instead to preserve the sparsity pattern of the tensor.
        if (static_cast<T>(value.second) == T(0)) {
            result.insert(coords, static_cast<T>(1));
        } else {
            result.insert(coords, static_cast<T>(value.second));
        }
    }
    result.pack();
    return result;
}

template<typename T, typename T2>
taco::Tensor<T> shiftLastMode(std::string name, taco::Tensor<T2> original) {
    taco::Tensor<T> result(name, original.getDimensions(), original.getFormat());
    std::vector<int> coords(original.getOrder());
    for (auto& value : taco::iterate<T2>(original)) {
        for (int i = 0; i < original.getOrder(); i++) {
            coords[i] = value.first[i];
        }
        int lastMode = original.getOrder() - 1;
        // For order 2 tensors, always shift the last coordinate. Otherwise, shift only coordinates
        // that have even last coordinates. This ensures that there is at least some overlap
        // between the original tensor and its shifted counter part.
        if (original.getOrder() <= 2 || (coords[lastMode] % 2 == 0)) {
            coords[lastMode] = (coords[lastMode] + 1) % original.getDimension(lastMode);
        }
        // TODO (rohany): Temporarily use a constant value here.
        result.insert(coords, T(2));
    }
    result.pack();
    return result;
}

// UfuncInputCache is a cache for the input to ufunc benchmarks. These benchmarks
// operate on a tensor loaded from disk and the same tensor shifted slightly. Since
// these operations are run multiple times, we can save alot in benchmark startup
// time from caching these inputs.
struct UfuncInputCache {
    template<typename U>
    std::pair<taco::Tensor<int64_t>, taco::Tensor<int64_t>> getUfuncInput(std::string path, U format, bool countNNZ = false, bool includeThird = false) {
        // See if the paths match.
        if (this->lastPath == path) {
            // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
            return std::make_pair(this->inputTensor, this->otherTensor);
        }

        // Otherwise, we missed the cache. Load in the target tensor and process it.

        this->lastLoaded = taco::read(path, format);

        // We assign lastPath after lastLoaded so that if taco::read throws an exception
        // then lastPath isn't updated to the new path.
        this->lastPath = path;
        this->inputTensor = castToType<int64_t>("A", this->lastLoaded);
        this->otherTensor = shiftLastMode<int64_t, int64_t>("B", this->inputTensor);
        if (countNNZ) {
            this->nnz = 0;
            for (auto& it : iterate<int64_t>(this->inputTensor)) {
                this->nnz++;
            }
        }
        if (includeThird) {
            this->thirdTensor = shiftLastMode<int64_t, int64_t>("C", this->otherTensor);
        }
        return std::make_pair(this->inputTensor, this->otherTensor);
    }

    taco::Tensor<double> lastLoaded;
    std::string lastPath;

    taco::Tensor<int64_t> inputTensor;
    taco::Tensor<int64_t> otherTensor;
    taco::Tensor<int64_t> thirdTensor;
    int64_t nnz;
};
UfuncInputCache inputCache;

const Format DSSS({Dense, Sparse, Sparse, Sparse}, {0,1,2,3});
const Format SSS({Sparse, Sparse, Sparse}, {0,1,2});
const Format DSS({Dense, Sparse, Sparse}, {0,1,2});

// vector<std::string> tensors3 = { "fb1k.tns"};
vector<std::string> tensors3 = { "facebook.tns", "fb1k.tns", "fb10k.tns",  "nell-1.tns", "nell-2.tns"};
// "amazon-reviews.tns", patents.tns", "reddit.tns" 


TEST(sam, pack_sss012) {
    std::string frosttPath = std::getenv("FROSTT_PATH");
    std::string frosttFormatPath = std::getenv("FROSTT_FORMATTED_TACO_PATH");
    for (auto& tnsPath : tensors3) {
        std::string frosttTensorPath = frosttPath;
        frosttTensorPath += "/"+tnsPath;

        auto pathSplit = taco::util::split(tnsPath, "/");
        auto filename = pathSplit[pathSplit.size() - 1];
        auto tensorName = taco::util::split(filename, ".")[0];
        cout << frosttTensorPath << endl;
        cout << tensorName << "..." << endl;

        Tensor<int64_t> frosttTensor, other;
        std::tie(frosttTensor, other) = inputCache.getUfuncInput(frosttTensorPath, SSS);

        ofstream origfile;
        std::string outpath = frosttFormatPath + "/";
        std::string origpath = outpath + tensorName + "_sss.txt";
        cout << origpath << endl;
        origfile.open (origpath);
        if(!origfile) {
            cout << "FAILED" << endl;
        }
        origfile << frosttTensor << endl;
        origfile.close();

        ofstream shiftfile;
        std::string shiftpath = outpath + tensorName + "_shift_sss.txt";
        cout << shiftpath << endl;
        shiftfile.open (shiftpath);
        if(!shiftfile) {
            cout << "FAILED" << endl;
        }
        shiftfile << other << endl;
        shiftfile.close();

    }
}

TEST(sam, pack_other_frostt) {
    std::string otherPath = std::getenv("TACO_TENSOR_PATH");
    otherPath += "/other";
    std::string otherFormattedPath = std::getenv("TACO_TENSOR_PATH");
    otherFormattedPath += "/other-formatted-taco";

    cout << otherPath << endl;

    vector<std::string> otherNames;

    for (auto& tnsPath : tensors3) {
        auto pathSplit = taco::util::split(tnsPath, "/");
        auto filename = pathSplit[pathSplit.size() - 1];
        auto tensorName = taco::util::split(filename, ".")[0];
        cout << tensorName << "..." << endl;

        if (std::experimental::filesystem::exists(otherPath)) {
            for (auto &entry: std::experimental::filesystem::directory_iterator(otherPath)) {
                std::string f(entry.path());

                // Check that the filename ends with .mtx.
                if (f.compare(f.size() - 4, 4, ".tns") == 0 && f.find(tensorName) != std::string::npos) {
                    otherNames.push_back(entry.path());
                }
            }
        }


        for (auto &otherfile : otherNames) {
            std::string filePath = otherfile;

            auto otherPathSplit = taco::util::split(otherfile, "/");
            cout << util::join(otherPathSplit) << endl;
            auto otherFilename = otherPathSplit[otherPathSplit.size() - 1];
            auto otherName = otherFilename.substr(0, otherFilename.size() - 4);
            
            cout << otherName << "..." << endl;

            Tensor<int64_t> tensor, other;
            Format format;
            if (otherName.find("vec") != std::string::npos) {
                format = Sparse;
            } else {
                format = DCSR;
            }

            std::tie(tensor, other) = inputCache.getUfuncInput(filePath, format);

            ofstream origfile;
            std::string outpath = otherFormattedPath + "/";
            std::string origpath = outpath + otherName + ".txt";
            origfile.open (origpath);
            if(!origfile) {
                cout << "FAILED" << endl;
            }
            origfile << tensor << endl;
            origfile.close();
        }
    }
}

TEST(sam, pack_other_ss) {
    std::string otherPath = std::getenv("TACO_TENSOR_PATH");
    otherPath += "/other";
    std::string otherFormattedPath = std::getenv("TACO_TENSOR_PATH");
    otherFormattedPath += "/other-formatted-taco";

    std::string tnsPath = getenv("SUITESPARSE_TENSOR_PATH");
    cout << otherPath << endl;

    vector<std::string> otherNames;

    auto pathSplit = taco::util::split(tnsPath, "/");
    auto filename = pathSplit[pathSplit.size() - 1];
    auto tensorName = taco::util::split(filename, ".")[0];
    cout << tensorName << "..." << endl;

    if (std::experimental::filesystem::exists(otherPath)) {
        for (auto &entry: std::experimental::filesystem::directory_iterator(otherPath)) {
            std::string f(entry.path());

            // Check that the filename ends with .mtx.
            if (f.compare(f.size() - 4, 4, ".tns") == 0 && f.find(tensorName) != std::string::npos) {
                otherNames.push_back(entry.path());
            }
        }
    }

    for (auto &otherfile : otherNames) {
        std::string filePath = otherfile;

        auto otherPathSplit = taco::util::split(otherfile, "/");
        cout << util::join(otherPathSplit) << endl;
        auto otherFilename = otherPathSplit[otherPathSplit.size() - 1];
        auto otherName = otherFilename.substr(0, otherFilename.size() - 4);

        cout << otherName << "..." << endl;

        Tensor<int64_t> tensor, other;
        Format format;
        if (otherName.find("vec") != std::string::npos) {
            format = Sparse;
        } else {
            format = DCSR;
        }

        // std::tie(tensor, other) = inputCache.getUfuncInput(filePath, format);

        tensor = castToType<int64_t>("C", taco::read(filePath, format));
        // Make sure tensor isn't empty
        if (tensor.getOrder() == 0) {
            if (otherName.find("vec") != std::string::npos) {
                cout << "---EMPTY: " << otherName << endl;
                tensor = Tensor<int64_t>("other", {1});
                tensor.insert({0}, int64_t(1));
            }
            else {
                tensor = Tensor<int64_t>("other", {1, 1});
                tensor.insert({0, 0}, int64_t(1));
            }
        }

        ofstream origfile;
        std::string outpath = otherFormattedPath + "/";
        std::string origpath = outpath + otherName + ".txt";
        cout << origpath << endl;
        origfile.open (origpath);
        if(!origfile) {
            cout << "FAILED" << endl;
        }
        origfile << tensor << endl;
        origfile.close();
    }
}
