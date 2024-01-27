/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#include "selective_scan_fwd_kernel.cuh"
#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);
 
template void selective_scan_bwd_cuda<float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<at::Half, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<at::BFloat16, float>(SSMParamsBwd &params, cudaStream_t stream);


