#ifndef GPU_KERNEL_HIP
#define GPU_KERNEL_HIP

#include "common.hpp"
#include "ring_buffer.hip.h"

__global__ void gpu_issue_batched_commands(DeviceToHostCmdBuffer* rbs);

#endif  // GPU_KERNEL_HIP