#ifndef TACO_SAM_GRAPH_H
#define TACO_SAM_GRAPH_H

#include <memory>
#include <vector>
#include <map>
#include "lower/iteration_graph.h"
#include "sam_nodes.h"
#include "taco/format.h"
#include "sam_ir.h"


namespace taco {

    class TensorVar;

    class IndexVar;

    class IndexExpr;

    class Assignment;

    class TensorPath;

    struct SAMGraphComputeNode {
        sam::SamNodeType nodeType;
        TensorVar tensor;
        int id;
        SAMGraphComputeNode *op1 = nullptr;
        SAMGraphComputeNode *op2 = nullptr;
        SAMGraphComputeNode *parent = nullptr;
    };

    void printSAMComputeNode (SAMGraphComputeNode node)
    {
        std::cout << node.id << ": ";
        switch (node.nodeType) {
            case sam::SamNodeType::Mul:
                std::cout << "type = Mul";
                break;
            case sam::SamNodeType::Add:
                std::cout << "type = Add";
                break;
            case sam::SamNodeType::Reduce:
                std::cout << "type = Reduce";
                break;
            case sam::SamNodeType::SparseAccumulator:
                std::cout << "type = SparseAccumulator";
                break;
            case sam::SamNodeType::Array:
                std::cout <<  node.tensor.getName();
                break;
            default:
                std::cout << "none";
                break;
        }
        std::cout << std::endl;
        if (node.op1 != nullptr) {
            std::cout << "child1: ";
            printSAMComputeNode(*node.op1);
        }
        if (node.op2 != nullptr)
        {
            std::cout << "child2: ";
            printSAMComputeNode(*node.op2);
        }

    }

/// An SAM graph consists of
    class SAMGraph {
    public:
        SAMGraph();

        /// Creates an iteration graph for a tensor with a defined expression.
        static SAMGraph make(Assignment);

        /// Returns the iteration graph roots; the index variables with no parents.
        const std::vector<IndexVar> &getRoots() const;

        /// Returns the children of the index variable
        const std::vector<IndexVar> &getChildren(const IndexVar &) const;

        /// Returns the parent of the index variable
        const IndexVar &getParent(const IndexVar &) const;

        /// Returns the ancestors of the index variable including itself.
        std::vector<IndexVar> getAncestors(const IndexVar &) const;

        /// Returns all descendant of the index variable, including itself.
        std::vector<IndexVar> getDescendants(const IndexVar &) const;


        /// Returns the tensor paths of the operand tensors in the iteration graph.
        const std::vector<TensorPath> &getTensorPaths() const;

        /// Returns the tensor path corresponding to a tensor read expression.
        const TensorPath &getTensorPath(const IndexExpr &) const;

        /// Returns the tensor path of the result tensor.
        const TensorPath &getResultTensorPath() const;


        /// Returns the index variable type.
        IndexVarType getIndexVarType(const IndexVar &) const;

        /// Returns true iff the index variable is free.
        bool isFree(const IndexVar &) const;

        /// Returns true iff the index variable is a reduction.
        bool isReduction(const IndexVar &) const;

        /// Returns true if the index variable is the only free var in its subtree.
        bool isLastFreeVariable(const IndexVar &) const;

        /// Returns true if the index variable is the ancestor of any free variable.
        bool hasFreeVariableDescendant(const IndexVar &) const;

        /// Returns true if the index variable has a reduction variable ancestor.
        bool hasReductionVariableAncestor(const IndexVar &) const;

        /// Returns the index expression at the given index variable.
        const IndexExpr &getIndexExpr(const IndexVar &) const;

        /// Print an SAM graph as a dot file.
        void printAsDot(std::ostream &);

        /// SAM Only Below, Iteration Graph Above (Should make SAM Graph a derived class of IterationGraph)

        /// Returns the iteration graph roots; the index variables with no parents.
        std::vector<IndexVar> getIndexVars() const;

        std::vector<IndexVar> getOrderedIndexVars() const;


        std::map<IndexVar, std::vector<TensorVar>> getContractions() const;

        /// Print the contraction map for the SAM graph
        void printContractions(std::ostream &os) const;

        /// Print the output iteration path (Fiber writes) for the results of the SAM graph
        void printOutputIteration(std::ostream &os);

        /// Get a list of index expression nodes of what the computation pattern will be
        std::vector<sam::SamNodeType> getComputation() const;

        /// Print the computation (ops, reductions) of the SAM graph
        void printComputation(std::ostream &os) const;

        static std::map<IndexVar, ModeFormat> getFormatMapping(const TensorPath path);

        /// Print the input iteration path (Fiber lookups and repeats) for the SAM graph
        void printInputIteration(std::ostream &os);

        /// Generate a SAM graph as a dot file for json export.
        void generateDotJSON(std::ostream &);

        void printInputIterationAsDot(std::ostream& os);

        /// Given an index variable and tensor path, gets the indexvariables mode for that tensor path
        size_t getMode(const IndexVar &, const TensorPath &) const;

        sam::SamIR makeGraph();



            /// Print a SAM graph.
        friend std::ostream &operator<<(std::ostream &, const SAMGraph &);

    private:
        struct Content;
        std::shared_ptr<Content> content;

    };
}

#endif //TACO_SAM_GRAPH_H
