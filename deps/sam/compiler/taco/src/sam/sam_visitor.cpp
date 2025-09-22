#include "sam_visitor.h"
#include "sam_nodes.h"

using namespace std;

namespace taco {
namespace sam {

// class IndexNotationVisitorStrict
    SAMVisitorStrict::~SAMVisitorStrict() {
    }

    SAMVisitor::~SAMVisitor() {
    }

    void SAMVisitorStrict::visit(const SamIR& sam) {
        sam.accept(this);
    }

    void SAMVisitor::visit(const RootNode *op) {
        for (auto node : op->nodes) {
            node.accept(this);
        }
    }

    void SAMVisitor::visit(const BroadcastNode *op) {
        for (auto node : op->outputs) {
            node.accept(this);
        }
    }

    void SAMVisitor::visit(const FiberLookupNode *op) {
        if (op->out_crd.defined()) {
            op->out_crd.accept(this);
        }

        if (op->out_ref.defined()) {
            op->out_ref.accept(this);
        }
    }

    void SAMVisitor::visit(const FiberWriteNode *op) {
    }


    void SAMVisitor::visit(const JoinerNode *op) {
        if (op->out_crd.defined()) {
            op->out_crd.accept(this);
        }

        for (auto out_ref : op->out_refs) {
            if (out_ref.defined()) {
                out_ref.accept(this);
            }
        }
    }

    void SAMVisitor::visit(const IntersectNode *op) {
        visit(static_cast<const JoinerNode*>(op));
    }

    void SAMVisitor::visit(const UnionNode *op) {
        visit(static_cast<const JoinerNode*>(op));
    }

    void SAMVisitor::visit(const RepeatNode *op) {
        if (op->out_ref.defined()) {
            op->out_ref.accept(this);
        }
    }

    void SAMVisitor::visit(const RepeatSigGenNode *op) {
        if (op->out_repsig.defined()) {
            op->out_repsig.accept(this);
        }
    }

    void SAMVisitor::visit(const ArrayNode *op) {
        if (op->out_val.defined()) {
            op->out_val.accept(this);
        }
    }

    void SAMVisitor::visit(const ComputeNode *op) {
        if (op->out_val.defined()) {
            op->out_val.accept(this);
        }
    }
    void SAMVisitor::visit(const MulNode *op) {
        visit(static_cast<const ComputeNode*>(op));
    }

    void SAMVisitor::visit(const AddNode *op) {
        visit(static_cast<const ComputeNode*>(op));
    }

    void SAMVisitor::visit(const ReduceNode *op) {
        visit(static_cast<const ComputeNode*>(op));
    }

    void SAMVisitor::visit(const SparseAccumulatorNode *op) {
        if (op->out_val.defined())
            op->out_val.accept(this);

        for (auto out_crd: op->out_crds) {
            out_crd.second.accept(this);
        }
    }

    void SAMVisitor::visit(const CrdDropNode *op) {
        if (op->out_outer_crd.defined())
            op->out_outer_crd.accept(this);

        if (op->out_inner_crd.defined())
            op->out_inner_crd.accept(this);

    }

    void SAMVisitor::visit(const CrdHoldNode *op) {
        if (op->out_outer_crd.defined())
            op->out_outer_crd.accept(this);

        if (op->out_inner_crd.defined())
            op->out_inner_crd.accept(this);

    }
}
}