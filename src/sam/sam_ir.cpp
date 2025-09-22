//
// Created by oliviahsu on 4/13/22.
//

#include "sam_nodes.h"
#include "sam_ir.h"


using namespace std;
namespace taco {
namespace sam {

    void SamIR::accept(SAMVisitorStrict *v) const {
        ptr->accept(v);
    }

    string SamIR::getTensorName() const {
        return ptr->getTensorName();
    }

    std::ostream& operator<<(std::ostream &os, const taco::sam::SamIR& sam) {
        os << sam.ptr->getName();
        return os;
    }

    // FiberLookup
    FiberLookup::FiberLookup(const FiberLookupNode *n) : SamIR(n) {
    }

    FiberLookup::FiberLookup(SamIR out_ref, SamIR out_crd, IndexVar i,
                             const TensorVar& tensorVar, int mode, int nodeID, bool root, bool source,
                             bool printEdgeName) :
    FiberLookup(new FiberLookupNode(out_ref, out_crd, i, tensorVar, mode, root, source, printEdgeName, nodeID)) {}

    FiberLookup::FiberLookup(SamIR out_ref, SamIR out_crd, IndexVar i,
                             const TensorVar& tensorVar, int mode, int nodeID, std::map<SamIR, std::string> edgeName,
                             bool root, bool source) :
            FiberLookup(new FiberLookupNode(out_ref, out_crd, i, tensorVar, mode, root, source, true,
                                            edgeName, nodeID)) {}

    
    TensorVar FiberLookup::getTensorVar() const {
        return getNode(*this)->tensorVar;
    }

    IndexVar FiberLookup::getIndexVar() const {
        return getNode(*this)->i;
    }

    template <> bool isa<FiberLookup>(SamIR s) {
        return isa<FiberLookupNode>(s.ptr);
    }

    template <> FiberLookup to<FiberLookup>(SamIR s) {
        taco_iassert(isa<FiberLookup>(s));
        return FiberLookup(to<FiberLookupNode>(s.ptr));
    }
    
    // FiberWrite
    FiberWrite::FiberWrite(const FiberWriteNode *n) : SamIR(n) {}

    FiberWrite::FiberWrite(IndexVar i, const TensorVar& tensorVar, int mode, string maxSegSize, string maxCrdSize,
                           int nodeID, bool sink, bool vals) :
    FiberWrite(new FiberWriteNode(i, tensorVar, mode, maxSegSize, maxCrdSize, sink, vals, nodeID)) {}

    template <> bool isa<FiberWrite>(SamIR s) {
        return isa<FiberWriteNode>(s.ptr);
    }

    template <> FiberWrite to<FiberWrite>(SamIR s) {
        taco_iassert(isa<FiberWrite>(s));
        return FiberWrite(to<FiberWriteNode>(s.ptr));
    }
    
    // Repeat
    Repeat::Repeat(const RepeatNode *n) : SamIR(n){
    }

    Repeat::Repeat(SamIR out_ref, IndexVar i, const TensorVar& tensorVar, int nodeID, bool root) :
    Repeat(new RepeatNode(out_ref, i, tensorVar, root, nodeID)) {
    }

    template <> bool isa<Repeat>(SamIR s) {
        return isa<RepeatNode>(s.ptr);
    }

    template <> Repeat to<Repeat>(SamIR s) {
        taco_iassert(isa<Repeat>(s));
        return Repeat(to<RepeatNode>(s.ptr));
    }
    
    // Intersect
    Intersect::Intersect(const IntersectNode *n) : SamIR(n){
    }

    Intersect::Intersect(SamIR out_crd, vector<SamIR> out_refs, IndexVar i, int nodeID, bool printEdgeName,
                         string edgeName) :
            Intersect(new IntersectNode(out_crd, out_refs, i, printEdgeName, edgeName,
                                        nodeID)) {
    }

    template <> bool isa<Intersect>(SamIR s) {
        return isa<IntersectNode>(s.ptr);
    }

    template <> Intersect to<Intersect>(SamIR s) {
        taco_iassert(isa<Intersect>(s));
        return Intersect(to<IntersectNode>(s.ptr));
    }
    
    // Union
    Union::Union(const UnionNode *n) : SamIR(n){
    }

    Union::Union(SamIR out_crd, vector<SamIR> out_refs, IndexVar i, int nodeID, bool printEdgeName, string edgeName) :
            Union(new UnionNode(out_crd, out_refs, i, printEdgeName, edgeName, nodeID)) {
    }

    template <> bool isa<Union>(SamIR s) {
        return isa<UnionNode>(s.ptr);
    }

    template <> Union to<Union>(SamIR s) {
        taco_iassert(isa<Union>(s));
        return Union(to<UnionNode>(s.ptr));
    }
    
    // Root
    Root::Root(const RootNode *n) : SamIR(n){
    }

    Root::Root(const vector<SamIR>& nodes, const vector<TensorVar>& tensors) : Root(new RootNode(nodes, tensors)){
    }
    
    template <> bool isa<Root>(SamIR s) {
        return isa<RootNode>(s.ptr);
    }

    template <> Root to<Root>(SamIR s) {
        taco_iassert(isa<Root>(s));
        return Root(to<RootNode>(s.ptr));
    }

    // Repeat Signal Generator
    RepeatSigGen::RepeatSigGen(const RepeatSigGenNode *n) : SamIR(n){
    }

    RepeatSigGen::RepeatSigGen(SamIR out_repsig, IndexVar i, int nodeID) :
            RepeatSigGen(new RepeatSigGenNode(out_repsig, i, nodeID)){
    }

    template <> bool isa<RepeatSigGen>(SamIR s) {
        return isa<RepeatSigGenNode>(s.ptr);
    }

    template <> RepeatSigGen to<RepeatSigGen>(SamIR s) {
        taco_iassert(isa<RepeatSigGen>(s));
        return RepeatSigGen(to<RepeatSigGenNode>(s.ptr));
    }
    
    // Broadcast
    Broadcast::Broadcast(const BroadcastNode *n) : SamIR(n) {
    }

    Broadcast::Broadcast(std::vector<SamIR> outputs, SamEdgeType type, int nodeID, bool printEdgeName) :
            Broadcast(new BroadcastNode(outputs, type, printEdgeName,
                                        nodeID)){

    }

    Broadcast::Broadcast(std::vector<SamIR> outputs, SamEdgeType type, int nodeID,
                         std::map<SamIR, std::string> edgeName) :
    Broadcast(new BroadcastNode(outputs, type, true, edgeName,
                                nodeID)){

    }

    template <> bool isa<Broadcast>(SamIR s) {
        return isa<BroadcastNode>(s.ptr);
    }

    template <> Broadcast to<Broadcast>(SamIR s) {
        taco_iassert(isa<Broadcast>(s));
        return Broadcast(to<BroadcastNode>(s.ptr));
    }
    
    // Array
    Array::Array(const ArrayNode *n) : SamIR(n){
    }

    Array::Array(SamIR out_val, const TensorVar& tensorVar, int nodeID, bool printEdgeName, bool root) :
    Array(new ArrayNode(out_val, tensorVar, printEdgeName, root, nodeID)) {
    }

    template <> bool isa<Array>(SamIR s) {
        return isa<ArrayNode>(s.ptr);
    }

    template <> Array to<Array>(SamIR s) {
        taco_iassert(isa<Array>(s));
        return Array(to<ArrayNode>(s.ptr));
    }
    
    // Mul 
    Mul::Mul(const MulNode *n) : SamIR(n){
    }

    Mul::Mul(SamIR out_val, int nodeID) :
            Mul(new MulNode(out_val, nodeID)) {
    }

    template <> bool isa<Mul>(SamIR s) {
        return isa<MulNode>(s.ptr);
    }

    template <> Mul to<Mul>(SamIR s) {
        taco_iassert(isa<Mul>(s));
        return Mul(to<MulNode>(s.ptr));
    }
    
    // Add
    Add::Add(const AddNode *n) : SamIR(n){
    }

    Add::Add(SamIR out_val, int nodeID, bool sub) :
            Add(new AddNode(out_val, sub, nodeID)) {
    }

    template <> bool isa<Add>(SamIR s) {
        return isa<AddNode>(s.ptr);
    }

    template <> Add to<Add>(SamIR s) {
        taco_iassert(isa<Add>(s));
        return Add(to<AddNode>(s.ptr));
    }
    
    // Reduce (Inner Reduction)
    Reduce::Reduce(const ReduceNode *n) : SamIR(n){
    }

    Reduce::Reduce(SamIR out_val, int nodeID) :
            Reduce(new ReduceNode(out_val, nodeID)) {
    }

    template <> bool isa<Reduce>(SamIR s) {
       return isa<ReduceNode>(s.ptr);
    }

    template <> Reduce to<Reduce>(SamIR s) {
        taco_iassert(isa<Reduce>(s));
        return Reduce(to<ReduceNode>(s.ptr));
    }
    
    // Sparse Accumulator (Outer reduction)
    SparseAccumulator::SparseAccumulator(const SparseAccumulatorNode *n) : SamIR(n){
    }

    SparseAccumulator::SparseAccumulator(SamIR out_val, map<int, SamIR> out_crds, int order,
                                         std::map<int, IndexVar> ivarMap, int nodeID) :
            SparseAccumulator(new SparseAccumulatorNode(out_val, out_crds, order, ivarMap, nodeID)) {
    }

    template <> bool isa<SparseAccumulator>(SamIR s) {
        return isa<SparseAccumulatorNode>(s.ptr);
    }

    template <> SparseAccumulator to<SparseAccumulator>(SamIR s) {
        taco_iassert(isa<SparseAccumulator>(s));
        return SparseAccumulator(to<SparseAccumulatorNode>(s.ptr));
    }

    // Crd Drop
    CrdDrop::CrdDrop(const CrdDropNode *n) : SamIR(n){
    }

    CrdDrop::CrdDrop(SamIR out_outer_crd, SamIR out_inner_crd, IndexVar outer, IndexVar inner, int nodeID) :
            CrdDrop(new CrdDropNode(out_outer_crd, out_inner_crd, outer, inner, nodeID)) {
    }
    
    template <> bool isa<CrdDrop>(SamIR s) {
        return isa<CrdDropNode>(s.ptr);
    }

    template <> CrdDrop to<CrdDrop>(SamIR s) {
        taco_iassert(isa<CrdDrop>(s));
        return CrdDrop(to<CrdDropNode>(s.ptr));
    }

    // Crd Hold
    CrdHold::CrdHold(const CrdHoldNode *n) : SamIR(n){
    }

    CrdHold::CrdHold(SamIR out_outer_crd, SamIR out_inner_crd, IndexVar outer, IndexVar inner, int nodeID) :
            CrdHold(new CrdHoldNode(out_outer_crd, out_inner_crd, outer, inner, nodeID)) {
    }

    template <> bool isa<CrdHold>(SamIR s) {
        return isa<CrdHoldNode>(s.ptr);
    }

    template <> CrdHold to<CrdHold>(SamIR s) {
        taco_iassert(isa<CrdHold>(s));
        return CrdHold(to<CrdHoldNode>(s.ptr));
    }
}
}



