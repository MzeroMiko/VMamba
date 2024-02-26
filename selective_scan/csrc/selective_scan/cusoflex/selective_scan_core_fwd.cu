/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#include "selective_scan_fwd_kernel_oflex.cuh"

template void selective_scan_fwd_cuda<1, float, float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<1, at::Half, float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<1, at::BFloat16, float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<1, at::Half, float, at::Half>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<1, at::BFloat16, float, at::BFloat16>(SSMParamsBase &params, cudaStream_t stream);

