/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#include "selective_scan_bwd_kernel_oflex.cuh"

template void selective_scan_bwd_cuda<1, float, float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::Half, float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::BFloat16, float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::Half, float, at::Half>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::BFloat16, float, at::BFloat16>(SSMParamsBwd &params, cudaStream_t stream);

