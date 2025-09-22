#include <vector>
#include <algorithm>

#include "sam_nodes.h"

using namespace std;

namespace taco {
namespace sam {
    // FiberLookupNode
    FiberLookupNode::FiberLookupNode(const SamIR& out_ref, const SamIR& out_crd, const IndexVar& i,
                                     const TensorVar& tensorVar, int mode, bool root, bool source, bool printEdgeName,
                                     std::map<SamIR, std::string> edgeName, int nodeID)
            : SAMNode(), out_ref(out_ref), out_crd(out_crd), tensorVar(tensorVar), mode(mode), i(i), root(root),
            source(source), printEdgeName(printEdgeName), edgeName(edgeName), nodeID(nodeID) {
        taco_iassert(mode < tensorVar.getOrder());
        if (mode >= 0) {
            auto modeOrders = tensorVar.getFormat().getModeOrdering();
            auto it = std::find(modeOrders.begin(), modeOrders.end(), mode);
            auto loc = std::distance(modeOrders.begin(), it);

            modeFormat = tensorVar.getFormat().getModeFormats().at(loc);
        }
    }

    FiberLookupNode::FiberLookupNode(const SamIR& out_ref, const SamIR& out_crd, const IndexVar& i,
                                     const TensorVar& tensorVar, int mode, bool root, bool source, bool printEdgeName,
                                     int nodeID)
            : SAMNode(), out_ref(out_ref), out_crd(out_crd), tensorVar(tensorVar), mode(mode), i(i), root(root),
              source(source), printEdgeName(printEdgeName), nodeID(nodeID) {
        taco_iassert(mode < tensorVar.getOrder());
        if (mode >= 0) {
            auto modeOrders = tensorVar.getFormat().getModeOrdering();
            auto it = std::find(modeOrders.begin(), modeOrders.end(), mode);
            auto loc = std::distance(modeOrders.begin(), it);

            modeFormat = tensorVar.getFormat().getModeFormats().at(loc);
        }
    }

    std::string FiberLookupNode::getName() const {
        stringstream ss;
        ss << "FiberLookup " << i.getName() << ": " << tensorVar.getName() << to_string(mode) << "\\n" <<
        modeFormat.getName();
        return ss.str();
    }

    FiberWriteNode::FiberWriteNode(IndexVar& i, const TensorVar& tensorVar, int mode, std::string maxSegSize,
                                   std::string maxCrdSize, bool sink, bool vals, int nodeID)
    : SAMNode(), tensorVar(tensorVar), mode(mode), i(i), maxSegSize(maxSegSize), maxCrdSize(maxCrdSize),
    sink(sink), vals(vals), nodeID(nodeID) {
        taco_iassert(mode < tensorVar.getOrder());
        if (mode >= 0) {
            auto modeOrders = tensorVar.getFormat().getModeOrdering();
            auto it = std::find(modeOrders.begin(), modeOrders.end(), mode);
            auto loc = std::distance(modeOrders.begin(), it);

            modeFormat = tensorVar.getFormat().getModeFormats().at(loc);
        }
    }

    std::string FiberWriteNode::getName() const {
        stringstream ss;
        if (vals) {
            ss << "FiberWrite Vals: " << tensorVar.getName();
        } else {
            ss << "FiberWrite " << i.getName() << ": " << tensorVar.getName() << to_string(mode) << "\\n"
               << modeFormat.getName();
        }
        return ss.str();
    }

    std::string RepeatNode::getName() const {
        stringstream ss;
        ss << "Repeat " <<  i.getName()  << ": " << tensorVar.getName();
        return ss.str();
    }

    std::string JoinerNode::getName() const {
        stringstream ss;
        ss << getNodeName() << " " << i.getName();
        return ss.str();
    }

    std::string RepeatSigGenNode::getName() const {
        stringstream ss;
        ss << "RepeatSignalGenerator " << i.getName();
        return ss.str();
    }

    std::string ArrayNode::getName() const {
        stringstream ss;
        ss << "Array Vals: " << tensorVar.getName();
        return ss.str();
    }

    std::string SparseAccumulatorNode::getName() const {
        stringstream ss;
        ss << "SparseAccumulator " << to_string(order) << " ";
        return ss.str();
    }

    std::string CrdDropNode::getName() const {
        stringstream ss;
        ss << "CrdDrop " << outer.getName() << "," << inner.getName();
        return ss.str();
    }

    std::string CrdHoldNode::getName() const {
        stringstream ss;
        ss << "CrdHold " << outer.getName() << "," << inner.getName();
        return ss.str();
    }
}
}
