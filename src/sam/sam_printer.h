//
// Created by oliviahsu on 4/14/22.
//

#ifndef TACO_SAM_PRINTER_H
#define TACO_SAM_PRINTER_H

#include <ostream>
#include <map>
#include <vector>
#include "sam_visitor.h"

namespace taco {
namespace sam {
    class SAMPrinter : public SAMVisitor {
    public:
        explicit SAMPrinter(std::ostream &os) : os(os) {}

        void print(const SamIR &expr);

        using SAMVisitorStrict::visit;

        void visit(const RootNode *);

        void visit(const FiberLookupNode *) override;

//        void visit(const FiberWriteNode *);
//
//        void visit(const RepeatNode *);
//
//        void visit(const RepeatSigGenNode *);
//
//        void visit(const IntersectNode *);
//
//        void visit(const UnionNode *);
//
//        void visit(const ArrayNode *);
//
//        void visit(const AddNode *);
//
//        void visit(const MulNode *);
//
//        void visit(const ReduceNode *);
//
//        void visit(const SparseAccumulatorNode *);

        void visit(const CrdDropNode *);
        void visit(const CrdHoldNode *);

    private:
        std::ostream &os;
    };

    class SAMDotNodePrinter : public SAMVisitor {
    public:
        explicit SAMDotNodePrinter(std::ostream &os) : os(os) {}

        void print(const SamIR &sam);

        using SAMVisitor::visit;

        void visit(const RootNode *) override;

        void visit(const BroadcastNode *) override;

        void visit(const FiberLookupNode *) override;

        void visit(const FiberWriteNode *) override;

        void visit(const RepeatNode *) override;

        void visit(const RepeatSigGenNode *) override;

        void visit(const JoinerNode *) override;

        void visit(const ArrayNode *);

        void visit(const ComputeNode *) override;
        void visit(const AddNode *) override;

        void visit(const SparseAccumulatorNode *) override;

        void visit(const CrdDropNode *) override;

        void visit(const CrdHoldNode *) override;

        void setPrintAttributes(bool printAttributes);
    private:
        std::ostream &os;
        bool prettyPrint = true;
        bool printAttributes = true;
        std::string tab = "    ";
        std::string name = "SAM";
        std::vector<int> printedNodes;

        std::string printTensorFormats(const RootNode *op);
    };

    class SAMDotEdgePrinter : public SAMVisitor {
    public:
        explicit SAMDotEdgePrinter(std::ostream &os) : os(os) {}

        void print(const SamIR &sam);

        using SAMVisitor::visit;

        void visit(const RootNode *) override;

        void visit(const BroadcastNode *) override;

        void visit(const FiberLookupNode *) override;

        void visit(const FiberWriteNode *) override;

        void visit(const RepeatNode *) override;

        void visit(const RepeatSigGenNode *) override;

        void visit(const JoinerNode *) override;

        void visit(const ArrayNode *) override;

        void visit(const ComputeNode *) override;

        void visit(const CrdDropNode *) override;

        void visit(const SparseAccumulatorNode *) override;

        void visit(const CrdHoldNode *) override;

        void setPrintAttributes(bool printAttributes);
    private:
        std::string printerHelper();

        std::ostream &os;
        bool prettyPrint = true;
        bool printAttributes = true;
        std::string tab = "    ";
        std::string edgeType;
        std::string edgeIndex;
        std::vector<int> printedNodes;
        std::map<std::string, std::string> edgeStyle = {{"ref", " style=bold"}, {"crd", " style=dashed"},
                                                        {"repsig", " style=dotted"}, {"bv", " style=dashed"}, {"",""}};
        bool printComment = false;
        std::string comment;
    };
}
}


#endif //TACO_SAM_PRINTER_H
