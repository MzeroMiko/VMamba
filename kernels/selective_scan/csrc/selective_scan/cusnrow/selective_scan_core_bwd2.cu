/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#include "selective_scan_bwd_kernel_nrow.cuh"

template void selective_scan_bwd_cuda<2, float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<2, at::Half, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<2, at::BFloat16, float>(SSMParamsBwd &params, cudaStream_t stream);

