// session.cpp — Session base. Family code subclasses this; the base owns
// the loader and backend pointers so subclasses can call loader_->find(name)
// without re-implementing lifecycle management.

#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/weight_loader.h"

namespace audiocore {

// Loader/backend are protected members on Session; their construction is
// driven by subclasses' load() override. This file exists to anchor the
// vtable and provide a place for cross-cutting session helpers as they
// emerge (e.g. KV cache reset, sampling state).

}  // namespace audiocore
