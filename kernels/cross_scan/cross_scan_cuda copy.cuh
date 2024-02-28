#include <cub/config.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/block/block_raking_layout.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/config.cuh>
#include <cuda/std/type_traits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex
#include <cub/block/block_reduce.cuh>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdio.h>

int best_block_dim(int feat_dim){
    int best_dim;
    if (feat_dim < 384){
        best_dim = 64;
    }
    else{
        if (feat_dim < 1024){
            best_dim = 128;
        }
        else{
            best_dim = 256;
        }
    }
    return best_dim;
}

template <typename T>
__global__ void cross_scan_cuda_kernel(T* x, T* out, const int B, const int C, const int H, const int W) {
    // data: B, C, H, W
    // B: gridDim.z; H: gridDim.y; W: gridDim.x; C: threadIdx.x | i;
    int i0, o0, o1, o2, o3;
    int HW = H * W;
    int CHW = C * HW;
    int _4CHW = 4 * CHW;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        i0 = blockIdx.z * CHW + i * HW + blockIdx.y * W + blockIdx.x; // data position -> (h * W + w)
        o0 = blockIdx.z * _4CHW + 0 + i * HW + blockIdx.y * W + blockIdx.x; // data position -> (h * W + w)
        o1 = blockIdx.z * _4CHW + CHW + i * HW + blockIdx.x * H + blockIdx.y; // transpose H W -> (w * H + h)
        o2 = blockIdx.z * _4CHW + 2 * CHW + i * HW + HW - 1 - blockIdx.y * W - blockIdx.x; // flip H W
        o3 = blockIdx.z * _4CHW + 3 * CHW + i * HW + HW - 1 - blockIdx.x * H - blockIdx.y; // flip (trans H W)
        T data = (T)(__ldg(x + i0));
        out[o0] = data;
        out[o1] = data;
        out[o2] = data;
        out[o3] = data;
    }
}

template <typename T>
__global__ void cross_merge_cuda_kernel(T* x, T* out, const int B, const int C, const int H, const int W) {
    // data: B, C, H, W
    // B: gridDim.z; H: gridDim.y; W: gridDim.x; C: threadIdx.x | i;
    int i0, o0, o1, o2, o3;
    int HW = H * W;
    int CHW = C * HW;
    int _4CHW = 4 * CHW;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        i0 = blockIdx.z * CHW + i * HW + blockIdx.y * W + blockIdx.x; // data position -> (h * W + w)
        o0 = blockIdx.z * _4CHW + 0 + i * HW + blockIdx.y * W + blockIdx.x; // data position -> (h * W + w)
        o1 = blockIdx.z * _4CHW + CHW + i * HW + blockIdx.x * H + blockIdx.y; // transpose H W -> (w * H + h)
        o2 = blockIdx.z * _4CHW + 2 * CHW + i * HW + HW - 1 - blockIdx.y * W - blockIdx.x; // flip H W
        o3 = blockIdx.z * _4CHW + 3 * CHW + i * HW + HW - 1 - blockIdx.x * H - blockIdx.y; // flip (trans H W)
        out[i0] = (T)(__ldg(x + o0)) + (T)(__ldg(x + o1)) + (T)(__ldg(x + o2)) + (T)(__ldg(x + o3));
    }
}

// ================================

template <typename input_t>
void cross_scan_cuda(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream) {
    dim3 grid(W, H, B);
    dim3 block(best_block_dim(C));
    cross_scan_cuda_kernel<input_t><<<grid, block, 0, stream>>>(((input_t *)(x)), ((input_t *)(out)), B, C, H, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t>
void cross_merge_cuda(void *__restrict__ x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream) {
    dim3 grid(W, H, B);
    dim3 block(best_block_dim(C));
    cross_merge_cuda_kernel<input_t><<<grid, block, 0, stream>>>(((input_t *)(x)), ((input_t *)(out)), B, C, H, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

