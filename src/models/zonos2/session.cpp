#include "audiocore/models/zonos2/family.h"

#include <cstdio>
#include <string>

#include "ggml.h"

namespace audiocore::zonos2 {

Zonos2Session::Zonos2Session() = default;

Zonos2Session::~Zonos2Session() {
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

}  // namespace audiocore::zonos2
