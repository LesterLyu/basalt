#include "trace.hpp"
#include <nvtx3/nvToolsExt.h>

namespace basalt::deepseek41 {
void trace_push(const char* name) { nvtxRangePushA(name); }
void trace_pop() { nvtxRangePop(); }
}
