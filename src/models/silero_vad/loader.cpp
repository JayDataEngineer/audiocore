// loader.cpp — family registration for silero_vad.

#include "audiocore/models/silero_vad.h"
#include "audiocore/framework/runtime/registry.h"

namespace audiocore::silero_vad {

static std::unique_ptr<Session> make_session() {
    return std::make_unique<SileroVadSession>();
}

}  // namespace audiocore::silero_vad

AUDIOCORE_REGISTER_FAMILY(silero_vad, audiocore::silero_vad::make_session)
