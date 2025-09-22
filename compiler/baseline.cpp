// TACO code for CPU benchmarking SAM

#include <fstream>
// We're using c++14, so wer're stuck with experimental filesystem.
#include <experimental/filesystem>
#include <tuple>

#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/transformations.h"

using namespace taco;

#define DIM_EXTRA 10 

template<int I, class...Ts>
decltype(auto) get(Ts &&... ts) {
    return std::get<I>(std::forward_as_tuple(ts...));
}

template<class ...ExtraArgs>
void printTensor(TensorBase tensor, std::string location, std::string benchName, int dim, ExtraArgs &&... extra_args) {
    auto &sparsity = get<0>(extra_args...);
    auto opType = get<1>(extra_args...);

    std::string sparseStr = std::to_string(sparsity);
    sparseStr = sparseStr.substr(2, sparseStr.size());
    std::string filename = location + "/" + benchName + "_" + opType + "_" + \
                          std::to_string(dim) + "_" + sparseStr + "_" + tensor.getName() + ".txt";
    std::ofstream outfile(filename, std::ofstream::out);
    outfile << util::toString(tensor);
    outfile.close();
}

static void applyBenchSizes(benchmark::internal::Benchmark *b) {
    // b->ArgsProduct({{250, 500, 750, 1000, 2500, 5000, 7500, 8000}});
    b->ArgsProduct({{10}});
}

// TensorInputCache is a cache for the input to ufunc benchmarks. These benchmarks
// operate on a tensor loaded from disk and the same tensor shifted slightly. Since
// these operations are run multiple times, we can save alot in benchmark startup
// time from caching these inputs.
struct TensorInputCache {
    template<typename U>
    std::pair<taco::Tensor<int64_t>, taco::Tensor<int64_t>>
    getTensorInput(std::string path, std::string datasetName, U format, bool countNNZ = false, bool includeThird = false,
                   bool includeVec = false, bool includeMat = false, bool genOther = false) {
        // See if the paths match.
        if (this->lastPath == path and this->lastFormat == format) {
            // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
            return std::make_pair(this->inputTensor, this->otherTensor);
        }

        // Otherwise, we missed the cache. Load in the target tensor and process it.
        this->lastLoaded = taco::read(path, format);
        this->lastFormat = format;
        // We assign lastPath after lastLoaded so that if taco::read throws an exception
        // then lastPath isn't updated to the new path.
        this->lastPath = path;
        this->inputTensor = castToType<int64_t>("B", this->lastLoaded);
        this->otherTensor = shiftLastMode<int64_t, int64_t>("C", this->inputTensor);

        if (countNNZ) {
            this->nnz = 0;
            for (auto &it: iterate<int64_t>(this->inputTensor)) {
                this->nnz++;
            }
        }
        if (includeThird) {
            this->thirdTensor = shiftLastMode<int64_t, int64_t>("D", this->otherTensor);
            this->otherTensorTrans = this->otherTensor.transpose("C", {1, 0}, DCSC);

        }
        if (includeVec and genOther) {
            this->otherVecFirstMode = genOtherVec<int64_t, int64_t>("C", datasetName, this->inputTensor);
            auto lastMode = this->inputTensor.getDimensions().size() - 1;
            this->otherVecLastMode = genOtherVec<int64_t, int64_t>("D", datasetName, this->inputTensor, lastMode);
        } else if (includeVec) {
            std::vector<int32_t> firstDim;
            std::vector<int32_t> lastDim;
            if (this->inputTensor.getOrder() == 2) {
                firstDim.push_back(this->inputTensor.getDimension(0));
                lastDim.push_back(this->inputTensor.getDimension(1));
            } else {
                firstDim.push_back(this->inputTensor.getDimension(0));
                lastDim.push_back(this->inputTensor.getDimension(2));
            }

            this->otherVecFirstMode = getOtherVec<int64_t, int64_t>("C", datasetName, this->inputTensor, firstDim);
            auto lastMode = this->inputTensor.getDimensions().size() - 1;
            this->otherVecLastMode = getOtherVec<int64_t, int64_t>("D", datasetName, this->inputTensor, lastDim, lastMode);
        }

        if (this->inputTensor.getOrder() > 2 and includeMat and genOther) {
            int DIM1 = this->inputTensor.getDimension(1);
            int DIM2 = this->inputTensor.getDimension(2);

            this->otherMatTTM = genOtherMat<int64_t, int64_t>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "ttm", 2);
            this->otherMatMode1MTTKRP = genOtherMat<int64_t, int64_t>("D", datasetName, this->inputTensor, {DIM_EXTRA, DIM1}, "mttkrp", 1);
            this->otherMatMode2MTTKRP = genOtherMat<int64_t, int64_t>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "mttkrp", 2);
        } else if (this->inputTensor.getOrder() > 2 and includeMat) {
            int DIM1 = this->inputTensor.getDimension(1);
            int DIM2 = this->inputTensor.getDimension(2);

            this->otherMatTTM = getOtherMat<int64_t, int64_t>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "ttm", 2);
            this->otherMatMode1MTTKRP = getOtherMat<int64_t, int64_t>("D", datasetName, this->inputTensor, {DIM_EXTRA, DIM1}, "mttkrp", 1);
            this->otherMatMode2MTTKRP = getOtherMat<int64_t, int64_t>("C", datasetName, this->inputTensor, {DIM_EXTRA, DIM2}, "mttkrp", 2);
        }
        return std::make_pair(this->inputTensor, this->otherTensor);
    }

    taco::Tensor<double> lastLoaded;
    std::string lastPath;
    taco::Format lastFormat;

    taco::Tensor<int64_t> inputTensor;
    taco::Tensor<int64_t> otherTensor;
    taco::Tensor<int64_t> thirdTensor;
    taco::Tensor<int64_t> otherTensorTrans;
    taco::Tensor<int64_t> otherVecFirstMode;
    taco::Tensor<int64_t> otherVecLastMode;

    // FROSTT only
    taco::Tensor<int64_t> otherMatTTM;
    taco::Tensor<int64_t> otherMatMode1MTTKRP;
    taco::Tensor<int64_t> otherMatMode2MTTKRP;

    int64_t nnz;
};

TensorInputCache inputCache;

std::string cpuBenchKey(std::string tensorName, std::string funcName) {
    return tensorName + "-" + funcName + "-taco";
}

enum FrosttOp {
    TTV = 1,
    TTM = 2,
    MTTKRP = 3,
    INNERPROD = 4,
    PLUS2 = 5
};

std::string opName(FrosttOp op) {
    switch (op) {
        case TTV: {
            return "ttv";
        }
        case TTM: {
            return "ttm";
        }
        case MTTKRP: {
            return "mttkrp";
        }
        case INNERPROD: {
            return "innerprod";
        }
        case PLUS2: {
            return "plus2";
        }
        default:
            return "";
        }
}

static void bench_frostt(benchmark::State &state, std::string tnsPath, FrosttOp op, int fill_value = 0) {
    bool GEN_OTHER = getEnvVar("GEN") == "ON";
    auto frosttTensorPath = getEnvVar("FROSTT_PATH");
    frosttTensorPath += "/" + tnsPath;

    auto pathSplit = taco::util::split(tnsPath, "/");
    auto filename = pathSplit[pathSplit.size() - 1];
    auto tensorName = taco::util::split(filename, ".")[0];
    state.SetLabel(tensorName);

    // TODO (rohany): What format do we want to do here?
    Tensor<int64_t> frosttTensor, otherShifted;

    std::tie(frosttTensor, otherShifted) = inputCache.getTensorInput(frosttTensorPath, tensorName, Sparse,
                                                                     false, false, true, true, GEN_OTHER);

    // std::cout << "Running benchmark tensor " << tnsPath << " on expression " << opName(op) << std::endl;

    int DIM0 = frosttTensor.getDimension(0);
    int DIM1 = frosttTensor.getDimension(1);
    int DIM2 = frosttTensor.getDimension(2);

    for (auto _: state) {
        state.PauseTiming();
        Tensor<int64_t> result;
        switch (op) {
            case TTV: {
                result = Tensor<int64_t>("result", {DIM0, DIM1}, DCSR, fill_value);

                Tensor<int64_t> otherVec = inputCache.otherVecLastMode;

                IndexVar i, j, k;
                result(i, j) = frosttTensor(i, j, k) * otherVec(k);
                break;
            }
            case INNERPROD: {
                result = Tensor<int64_t>("result");

                IndexVar i, j, k;
                result() = frosttTensor(i, j, k) * otherShifted(i, j, k);
                break;
            }
            case PLUS2: {
                result = Tensor<int64_t>("result", frosttTensor.getDimensions(), frosttTensor.getFormat(), fill_value);

                IndexVar i, j, k;
                result(i, j, k) = frosttTensor(i, j, k) + otherShifted(i, j, k);
                break;
            }
            case TTM: {
                // Assume otherMat(0) format is the same as frosttTensor(2)
                result = Tensor<int64_t>("result", {DIM0, DIM1, DIM_EXTRA}, frosttTensor.getFormat(), fill_value);
                Tensor<int64_t> otherMat = inputCache.otherMatTTM;

                IndexVar i, j, k, l;
                // TODO: (owhsu) need to pick things for this and MTTKRP...
                result(i, j, k) = frosttTensor(i, j, l) * otherMat(k, l);
                break;
            }
            case MTTKRP: {
                result = Tensor<int64_t>("result", {DIM0, DIM_EXTRA}, DCSR, fill_value);

                Tensor<int64_t> otherMat = inputCache.otherMatMode1MTTKRP;
                Tensor<int64_t> otherMat1 = inputCache.otherMatMode2MTTKRP;

                IndexVar i, j, k, l;
                result(i, j) = frosttTensor(i, k, l) * otherMat(j, k) * otherMat1(j, l);
                break;
            }
            default:
                state.SkipWithError("invalid expression");
                return;
        }
        result.setAssembleWhileCompute(true);
        result.compile();
        state.ResumeTiming();

        result.compute();

        state.PauseTiming();

        if (auto validationPath = getValidationOutputPath(); validationPath != "") {
            auto key = cpuBenchKey(tensorName, opName(op));
            auto outpath = validationPath + key + ".tns";
            taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
        }
        state.ResumeTiming();

    }
}

#define FOREACH_FROSTT_TENSOR(__func__) \
  __func__(facebook, "facebook.tns") \
  __func__(fb1k, "fb1k.tns") \
  __func__(fb10k, "fb10k.tns") \
  __func__(nell-1, "nell-1.tns") \
  __func__(nell-2, "nell-2.tns")

// Other FROSTT tensors that may or may not be too large to load.
// __func__(delicious, "delicious.tns") \
  // __func__(flickr, "flickr.tns") \
  // __func__(patents, "patents.tns") \
  // __func__(reddit, "reddit.tns") \
  // __func__(amazon-reviews, "amazon-reviews.tns") \
  // lbnl-network is fine, but python can't load it.
// __func__(lbnl-network, "lbnl-network.tns") \
 

#define DECLARE_FROSTT_BENCH(name, path) \
  TACO_BENCH_ARGS(bench_frostt, name/tensor3_ttv, path, TTV); \
  TACO_BENCH_ARGS(bench_frostt, name/tensor3_innerprod, path, INNERPROD); \
  TACO_BENCH_ARGS(bench_frostt, name/tensor3_elemadd_plus2, path, PLUS2); \
  TACO_BENCH_ARGS(bench_frostt, name/tensor3_ttm, path, TTM); \
  TACO_BENCH_ARGS(bench_frostt, name/tensor3_mttkrp, path, MTTKRP);


FOREACH_FROSTT_TENSOR(DECLARE_FROSTT_BENCH)

struct SuiteSparseTensors {
    SuiteSparseTensors() {
        auto ssTensorPath = getTacoTensorPath();
        ssTensorPath += "suitesparse/";
        if (std::experimental::filesystem::exists(ssTensorPath)) {
            for (auto &entry: std::experimental::filesystem::directory_iterator(ssTensorPath)) {
                std::string f(entry.path());
                // Check that the filename ends with .mtx.
                if (f.compare(f.size() - 4, 4, ".mtx") == 0) {
                    this->tensors.push_back(entry.path());
                }
            }
        }
    }

    std::vector<std::string> tensors;
};

SuiteSparseTensors ssTensors;

enum SuiteSparseOp {
    SPMV = 1,
    SPMM = 2,
    PLUS3 = 3,
    SDDMM = 4,
    MATTRANSMUL = 5,
    RESIDUAL = 6,
    MMADD = 7,
    MMMUL = 8
};

std::string opName(SuiteSparseOp op) {
    switch (op) {
        case SPMV: {
            return "spmv";
        }
        case SPMM: {
            return "spmm";
        }
        case PLUS3: {
            return "plus3";
        }
        case SDDMM: {
            return "sddmm";
        }
        case MATTRANSMUL: {
            return "mattransmul";
        }
        case RESIDUAL: {
            return "residual";
        }
        case MMADD: {
            return "mmadd";
        }
	case MMMUL: {
	    return "mmmul"
	}
        default:
            return "";
    }
}

static void bench_suitesparse(benchmark::State &state, SuiteSparseOp op, int fill_value = 0) {

    bool GEN_OTHER = getEnvVar("GEN") == "ON";

    // Counters must be present in every run to get reported to the CSV.
    state.counters["dimx"] = 0;
    state.counters["dimy"] = 0;
    state.counters["nnz"] = 0;
    state.counters["other_sparsity1"] = 0;
    state.counters["other_sparsity1"] = 0;

    auto tensorPath = getEnvVar("SUITESPARSE_TENSOR_PATH");
    // std::cout << "Running " << opName(op) << " " << tensorPath << std::endl;
    if (tensorPath == "") {
        state.error_occurred();
        return;
    }

    auto pathSplit = taco::util::split(tensorPath, "/");
    auto filename = pathSplit[pathSplit.size() - 1];
    auto tensorName = taco::util::split(filename, ".")[0];
    state.SetLabel(tensorName);

    taco::Tensor<int64_t> ssTensor, otherShifted;
    try {
        taco::Format format = op == MATTRANSMUL ? DCSC : DCSR;
        std::tie(ssTensor, otherShifted) = inputCache.getTensorInput(tensorPath, tensorName, format, true /* countNNZ */,
                                                                     true /* includeThird */, true, false, GEN_OTHER);
    } catch (TacoException &e) {
        // Counters don't show up in the generated CSV if we used SkipWithError, so
        // just add in the label that this run is skipped.
        std::cout << e.what() << std::endl;
        state.SetLabel(tensorName + "/SKIPPED-FAILED-READ");
        return;
    }

    int DIM0 = ssTensor.getDimension(0);
    int DIM1 = ssTensor.getDimension(1);
    
    state.counters["dimx"] = DIM0;
    state.counters["dimy"] = DIM1;
    state.counters["nnz"] = inputCache.nnz;

    taco::Tensor<int64_t> denseMat1;
    taco::Tensor<int64_t> denseMat2;
    taco::Tensor<int64_t> s1("s1"), s2("s2");
    s1.insert({}, int64_t(2));
    s2.insert({}, int64_t(2));
    if (op == SDDMM) {
        denseMat1 = Tensor<int64_t>("denseMat1", {DIM0, DIM_EXTRA}, Format({dense, dense}));
        denseMat2 = Tensor<int64_t>("denseMat2", {DIM_EXTRA, DIM1}, Format({dense, dense}, {1, 0}));

        // (owhsu) Making this dense matrices of all 1's
        for (int kk = 0; kk < DIM_EXTRA; kk++) {
            for (int ii = 0; ii < DIM0; ii++) {
                denseMat1.insert({ii, kk}, int64_t(1));
            }
            for (int jj = 0; jj < DIM1; jj++) {
                denseMat2.insert({kk, jj}, int64_t(1));
            }
        }
    }

    for (auto _: state) {
        state.PauseTiming();
        Tensor<int64_t> result;
        IndexStmt stmt;

        switch (op) {
            case SPMV: {
                result = Tensor<int64_t>("result", {DIM0}, Format(Sparse), fill_value);
                Tensor<int64_t> otherVec = inputCache.otherVecLastMode;

                IndexVar i, j, k;
                result(i) = ssTensor(i, j) * otherVec(j);
                break;
            }
            case SPMM: {
                result = Tensor<int64_t>("result", {DIM0, DIM0}, DCSR, fill_value);
                Tensor<int64_t> otherShiftedTrans = inputCache.otherTensorTrans;

                IndexVar i, j, k;
                result(i, j) = ssTensor(i, k) * otherShiftedTrans(k, j);
                
                stmt = result.getAssignment().concretize();
                stmt = reorderLoopsTopologically(stmt);
                // stmt = stmt.assemble(result.getAssignment().getLhs().getTensorVar(), taco::AssembleStrategy::Append);
                break;
            }
            case PLUS3: {
                result = Tensor<int64_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);
                Tensor<int64_t> otherShifted2 = inputCache.thirdTensor;

                IndexVar i, j, k, l;
                result(i, j) = ssTensor(i, j) + otherShifted(i, j) + otherShifted2(i, j);
                break;
            }
            case SDDMM: {
                result = Tensor<int64_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

                IndexVar i, j, k;
                result(i, j) = ssTensor(i, j) * denseMat1(i, k) * denseMat2(k, j);
                break;
            }
            case RESIDUAL: {
                result = Tensor<int64_t>("result", {DIM0}, Format(Sparse), fill_value);

                Tensor<int64_t> otherVeci = inputCache.otherVecFirstMode;
                Tensor<int64_t> otherVecj = inputCache.otherVecLastMode;

                IndexVar i, j, k;
                result(i) = otherVeci(i) - ssTensor(i, j) * otherVecj(j);
                break;
            }
            case MMADD: {
                result = Tensor<int64_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

                IndexVar i, j, k;
                result(i, j) = ssTensor(i, j) + otherShifted(i, j);
                break;
            }
            case MMMUL: {
                result = Tensor<int64_t>("result", ssTensor.getDimensions(), ssTensor.getFormat(), fill_value);

                IndexVar i, j, k;
                result(i, j) = ssTensor(i, j) * otherShifted(i, j);
                break;
            }
            case MATTRANSMUL: {
                result = Tensor<int64_t>("result", {DIM1}, Format(Sparse), fill_value);



                Tensor<int64_t> otherVeci = inputCache.otherVecLastMode;
                    Tensor<int64_t> otherVecj = inputCache.otherVecFirstMode;

                    IndexVar i, j;
                    result(i) = s1() * ssTensor(j, i) * otherVecj(j) + s2() * otherVeci(i);
                    break;
                }
                default:
                    state.SkipWithError("invalid expression");
                    return;
            }
            
            if (op == SPMM) {
                result.compile(stmt);
                state.ResumeTiming();
                result.assemble();
            }
            else {
                result.setAssembleWhileCompute(true);
                result.compile();
                state.ResumeTiming();
            }

            result.compute();

            state.PauseTiming();
            if (auto validationPath = getValidationOutputPath(); validationPath != "") {
                auto key = cpuBenchKey(tensorName, opName(op));
                auto outpath = validationPath + key + ".tns";
                taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
            }
            state.ResumeTiming();

        }
    }

    TACO_BENCH_ARGS(bench_suitesparse, vecmul_spmv, SPMV);
    TACO_BENCH_ARGS(bench_suitesparse, mat_elemadd3_plus3, PLUS3);
    TACO_BENCH_ARGS(bench_suitesparse, mat_sddmm, SDDMM);
    TACO_BENCH_ARGS(bench_suitesparse, mat_residual, RESIDUAL);
    TACO_BENCH_ARGS(bench_suitesparse, mat_elemadd_mmadd, MMADD);
    // TODO: need to fix for DCSC for this
    TACO_BENCH_ARGS(bench_suitesparse, mat_mattransmul, MATTRANSMUL);
    TACO_BENCH_ARGS(bench_suitesparse, matmul_spmm, SPMM);
    TACO_BENCH_ARGS(bench_suitesparse, mat_elemmul, MMMUL);

