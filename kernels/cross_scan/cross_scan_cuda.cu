#include "cross_scan_cuda.cuh"

template void cross_scan_cuda<float>(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);
template void cross_scan_cuda<at::Half>(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);
template void cross_scan_cuda<at::BFloat16>(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);
template void cross_merge_cuda<float>(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);
template void cross_merge_cuda<at::Half>(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);
template void cross_merge_cuda<at::BFloat16>(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);



