#ifndef TACO_SAM_NODES_H
#define TACO_SAM_NODES_H
#include <vector>
#include <memory>

#include "sam_visitor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "sam_nodes_abstract.h"
#include "sam_ir.h"


namespace taco {
namespace sam {

struct RootNode : public SAMNode {
    RootNode() : SAMNode() {}

    explicit RootNode(const std::vector<SamIR>& nodes, const std::vector<TensorVar>& tensors) :
    SAMNode(), nodes(nodes), tensors(tensors) {
    }

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override {
        return "RootNode";
    }

    std::vector<SamIR> nodes;
    std::vector<TensorVar> tensors;

    SamNodeType _type_info = SamNodeType::Root;
};


struct BroadcastNode : public SAMNode {
    BroadcastNode() : SAMNode() {}

    BroadcastNode(const std::vector<SamIR>& outputs, const SamEdgeType& type,
                  bool printEdgeName, int nodeID) :
            SAMNode(), outputs(outputs), type(type), printEdgeName(printEdgeName), nodeID(nodeID) {
    }

    BroadcastNode(const std::vector<SamIR>& outputs, const SamEdgeType& type,
                           bool printEdgeName, std::map<SamIR, std::string> edgeName,  int nodeID) :
            SAMNode(), outputs(outputs), type(type), printEdgeName(printEdgeName), edgeName(edgeName), nodeID(nodeID) {
    }

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override {
        return "WireBroadcast";
    }

    std::vector<SamIR> outputs;
    SamEdgeType type;

    // Needed for Sparse Accumulator
    bool printEdgeName = false;
    std::map<SamIR, std::string> edgeName;

    int nodeID = 0;

    SamNodeType _type_info = SamNodeType::Broadcast;
};

struct FiberLookupNode : public SAMNode {
    FiberLookupNode() : SAMNode() {}

    FiberLookupNode(const SamIR& out_ref, const SamIR& out_crd, const IndexVar& i,
                    const TensorVar& tensorVar, int mode, bool root, bool source, bool printEdgeName, int nodeID);

    FiberLookupNode(const SamIR& out_ref, const SamIR& out_crd, const IndexVar& i,
                    const TensorVar& tensorVar, int mode, bool root, bool source, bool printEdgeName,
                    std::map<SamIR, std::string> edgeName, int nodeID);

    void accept(SAMVisitorStrict* v) const override{
        v->visit(this);
    }

    std::string getName() const override;

    std::string getTensorName() const override {
        return tensorVar.getName();
    }

    //Outputs
    SamIR out_ref;
    SamIR out_crd;

    // Metadata
    TensorVar tensorVar;
    /// Mode format for this fiber lookup block
    ModeFormat modeFormat;
    int mode = 0;
    IndexVar i;

    /// Dimension of this fiber lookup
    int dim = 0;
    /// If this fiber lookup is a SAM graph source (i.e. stream data from a user-provided input)
    /// This is used in the AHA backend example for glb mode
    bool root = false;
    bool source = false;

    /// This is needed for joiner nodes
    /// Also Sparse Accumulator
    bool printEdgeName = false;
    std::map<SamIR, std::string> edgeName;

    int nodeID = 0;

    static const SamNodeType _type_info = SamNodeType::FiberLookup;

};

struct FiberWriteNode : public SAMNode {
    FiberWriteNode() : SAMNode() {}

    FiberWriteNode(IndexVar& i, const TensorVar& tensorVar, int mode, std::string maxSegSize, std::string maxCrdSize,
                   bool sink, bool vals, int nodeID);

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override;

    std::string getTensorName() const override {
        return tensorVar.getName();
    }

    // Outputs
    // TODO: figure out if write scanners should have outputs (for workspaces)

    // Metadata
    TensorVar tensorVar;
    int mode = 0;
    ModeFormat modeFormat;
    IndexVar i;
    std::string maxSegSize;
    std::string maxCrdSize;

    /// Dimension of this fiber lookup
    int dim = 0;
    /// If this fiber write is a SAM graph sink (i.e. stream data back to the end-user as final result)
    /// This is used in the AHA backend example for glb mode
    bool sink = true;
    bool root = false;
    bool vals = false;

    int nodeID = 0;

    static const SamNodeType _type_info = SamNodeType::FiberWrite;
};


struct RepeatNode : public SAMNode {
    RepeatNode() : SAMNode() {}

    RepeatNode(const SamIR& out_ref, const IndexVar& i, const TensorVar& tensorVar, bool root, int nodeID)
            : SAMNode(), out_ref(out_ref),
            i(i), tensorVar(tensorVar), root(root), nodeID(nodeID) {
    }


    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override;

    std::string getTensorName() const override {
        return tensorVar.getName();
    }

    // Outputs
    SamIR out_ref;

    // Metadata
    IndexVar i;
    TensorVar tensorVar;

    bool root = false;

    int nodeID = 0;

    static const SamNodeType _type_info = SamNodeType::Repeat;
};

struct JoinerNode : public SAMNode {
    virtual std::string getNodeName() const = 0;
    virtual std::string getNodeStr() const = 0;

    std::string getName() const override;

    // No Outputs
    SamIR out_crd;
    std::vector<SamIR> out_refs;

    // Metadata: None
    IndexVar i;
    int numInputs = 0;
    int nodeID = 0;

    // Needed for coordinate drops
    // Needed also for Sparse Accumulator Nodes
    bool printEdgeName = false;
    std::string edgeName;
protected:
    JoinerNode() : SAMNode() {}

    JoinerNode(SamIR& out_crd, std::vector<SamIR>& out_refs, IndexVar& i, bool printEdgeName, std::string edgeName,
               int nodeID) :
    SAMNode(), out_crd(out_crd), out_refs(out_refs), i(i), printEdgeName(printEdgeName), edgeName(edgeName),
    nodeID(nodeID) {
//        numInputs = (int)out_refs.size();
    }
};

struct IntersectNode : public JoinerNode {
    IntersectNode() : JoinerNode() {}

    IntersectNode(SamIR& out_crd, std::vector<SamIR>& out_refs, IndexVar& i, bool printEdgeName, std::string edgeName,
                  int nodeID) :
    JoinerNode(out_crd, out_refs, i, printEdgeName, edgeName, nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getNodeName() const override {
        return "intersect";
    }

    std::string getNodeStr() const override {
        return "Intersect";
    }
    static const SamNodeType _type_info = SamNodeType::Intersect;
};

struct UnionNode : public JoinerNode {
    UnionNode() : JoinerNode() {}

    UnionNode(SamIR& out_crd, std::vector<SamIR>& out_refs, IndexVar& i, bool printEdgeName, std::string edgeName,
              int nodeID) :
    JoinerNode(out_crd, out_refs, i, printEdgeName, edgeName, nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getNodeName() const override {
        return "union";
    }

    std::string getNodeStr() const override {
        return "Union";
    }
    static const SamNodeType _type_info = SamNodeType::Union;
};

struct RepeatSigGenNode : public SAMNode {
    RepeatSigGenNode() : SAMNode() {}

    RepeatSigGenNode(const SamIR& out_repsig, const IndexVar& i, int nodeID) :
    SAMNode(), out_repsig(out_repsig), i(i), nodeID(nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override;

    // No Outputs
    SamIR out_repsig;

    // Metadata
    IndexVar i;

    int nodeID = 0;

    static const SamNodeType _type_info = SamNodeType::RepeatSigGen;
};

struct ArrayNode : public SAMNode {
    ArrayNode() : SAMNode() {}

    ArrayNode(const SamIR& out_val, const TensorVar& tensorVar, bool printEdgeName, bool root, int nodeID) :
            SAMNode(), out_val(out_val), tensorVar(tensorVar), printEdgeName(printEdgeName), root(root), nodeID(nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override;

    std::string getTensorName() const override {
        return tensorVar.getName();
    }

    // No Outputs
    SamIR out_val;

    // Metadata
    TensorVar tensorVar;

    bool vals = true;
    bool printEdgeName = false;
    bool root = false;

    int nodeID = 0;

    static const SamNodeType _type_info = SamNodeType::Array;
};

struct ComputeNode : public SAMNode {
    virtual std::string getNodeName() const = 0;
    virtual std::string getNodeStr() const = 0;

    std::string getName() const override {
        return getNodeStr();
    }
    
    // No Outputs
    SamIR out_val;

    // Metadata
    bool parentSub = false;
    // None
    int nodeID = 0;
protected:
    ComputeNode() : SAMNode() {}

    ComputeNode(const SamIR& out_val, int nodeID) : SAMNode(), out_val(out_val), nodeID(nodeID) {}
};

struct MulNode : public ComputeNode {
    MulNode() : ComputeNode() {}
    
    MulNode(const SamIR& out_val, int nodeID) : ComputeNode(out_val, nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getNodeName() const override {
        return "mul";
    }

    std::string getNodeStr() const override {
        return "Mul";
    }

    static const SamNodeType _type_info = SamNodeType::Mul;
};

struct AddNode : public ComputeNode {
    AddNode() : ComputeNode() {}

    AddNode(const SamIR& out_val, bool sub, int nodeID) : ComputeNode(out_val, nodeID), sub(sub) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getNodeName() const override {
        return "add";
    }

    std::string getNodeStr() const override {
        return "Add";
    }

    bool sub = false;
    static const SamNodeType _type_info = SamNodeType::Add;
};

struct ReduceNode : public ComputeNode {
    ReduceNode() : ComputeNode() {}

    ReduceNode(const SamIR& out_val, int nodeID) : ComputeNode(out_val, nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getNodeName() const override {
        return "reduce";
    }

    std::string getNodeStr() const override {
        return "Reduce";
    }

    static const SamNodeType _type_info = SamNodeType::Reduce;
};

struct SparseAccumulatorNode : public SAMNode {
    SparseAccumulatorNode() : SAMNode() {}

    SparseAccumulatorNode(const SamIR& out_val, const std::map<int, SamIR> out_crds, int order,
                          std::map<int, IndexVar> ivarMap, int nodeID) :
    SAMNode(), out_val(out_val), out_crds(out_crds), order(order), ivarMap(ivarMap), nodeID(nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getNodeName() const {
        return "spaccumulator";
    }
    std::string getName() const override;

    // Outputs
    SamIR out_val;
    std::map<int, SamIR> out_crds;

    // Metadata
    int order = 0;
    std::map<int, IndexVar> ivarMap;

    int nodeID = 0;
    static const SamNodeType _type_info = SamNodeType::SparseAccumulator;
};

struct CrdDropNode : public SAMNode {
    CrdDropNode() : SAMNode() {}

    CrdDropNode(const SamIR& out_outer_crd, const SamIR& out_inner_crd,
                const IndexVar& outer, const IndexVar& inner, int nodeID) :
            SAMNode(), out_outer_crd(out_outer_crd), out_inner_crd(out_inner_crd),
            outer(outer), inner(inner), nodeID(nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override;

    SamIR out_outer_crd;
    SamIR out_inner_crd;

    // Metadata
    IndexVar outer;
    IndexVar inner;

    int nodeID = 0;

    static const SamNodeType _type_info = SamNodeType::CrdDrop;
};

struct CrdHoldNode : public SAMNode {
    CrdHoldNode() : SAMNode() {}

    CrdHoldNode(const SamIR& out_outer_crd, const SamIR& out_inner_crd,
                const IndexVar& outer, const IndexVar& inner, int nodeID) :
            SAMNode(), out_outer_crd(out_outer_crd), out_inner_crd(out_inner_crd),
            outer(outer), inner(inner), nodeID(nodeID) {}

    void accept(SAMVisitorStrict* v) const override {
        v->visit(this);
    }

    std::string getName() const override;

    // No Outputs
    SamIR out_outer_crd;
    SamIR out_inner_crd;

    // Metadata
    IndexVar outer;
    IndexVar inner;

    int nodeID = 0;

    static const SamNodeType _type_info = SamNodeType::CrdDrop;
};

/// Returns true if expression e is of type E.
template <typename E>
inline bool isa(const SAMNode* e) {
    return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}

/// Casts the expression e to type E.
template <typename E>
inline const E* to(const SAMNode* e) {
    taco_iassert(isa<E>(e)) <<
                            "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
    return static_cast<const E*>(e);
}

template <typename I>
inline const typename I::Node* getNode(const I& sam) {
    taco_iassert(isa<typename I::Node>(sam.ptr));
    return static_cast<const typename I::Node*>(sam.ptr);
}

}
}


#endif //TACO_SAM_NODES_H
