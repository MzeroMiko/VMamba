/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#include "selective_scan_fwd_kernel.cuh"

template void selective_scan_fwd_cuda<3, float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<3, at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<3, at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);

