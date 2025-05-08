
#ifndef ROCPROFILER_WRAPPER_H
#define ROCPROFILER_WRAPPER_H

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

// Include the main ROCProfiler header
// This should bring in all the necessary HSA dependencies
#include <rocprofiler/rocprofiler.h>
#include "activity.h"

#endif // ROCPROFILER_WRAPPER_H