#ifndef TACO_SAM_NODES_ABSTRACT_H
#define TACO_SAM_NODES_ABSTRACT_H
#include <vector>
#include <memory>

#include "taco/type.h"
#include "taco/util/uncopyable.h"
#include "taco/util/intrusive_ptr.h"

namespace taco {
namespace sam {

class SAMVisitorStrict;
class SamIR;

enum class SamNodeType {
    Root,
    Broadcast,
    FiberLookup,
    FiberWrite,
    Repeat,
    RepeatSigGen,
    Intersect,
    Union,
    Array,
    Mul,
    Add,
    Reduce,
    SparseAccumulator,
    CrdDrop,
    CrdHold,
};

enum class SamEdgeType {
    ref,
    val,
    crd,
    repsig
};

//FIXME: have this abstract node initialize type_info and nodeID
struct SAMNode : public util::Manageable<SAMNode>,
                 private util::Uncopyable {
public:
    SAMNode() = default;

    virtual ~SAMNode() = default;

    virtual void accept(SAMVisitorStrict *) const = 0;

    virtual std::string getName() const = 0;

    virtual std::string getTensorName() const {
        return "";
    }

    SamNodeType type_info() const;

    bool printed = false;

protected:
    SamNodeType _type_info;
};

}
}
#endif //TACO_SAM_NODES_ABSTRACT_H
