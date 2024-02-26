/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, bool kIsEvenLen_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int MaxDState = MAX_DSTATE;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.n_chunks * params.dstate;

    float D_val = 0; // attention!
    if (params.D_ptr != nullptr) {
        D_val = reinterpret_cast<float *>(params.D_ptr)[dim_id];
    }
    float delta_bias = 0;
    if (params.delta_bias_ptr != nullptr) {
        delta_bias = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];
    }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNItems], delta_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);
        if constexpr (!kDirectIO) { __syncthreads(); }
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNItems], delta_u_vals[kNItems], out_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            float u_val = float(u_vals[i]);
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            if (params.delta_softplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
            }
            delta_u_vals[i] = delta_vals[i] * u_val;
            out_vals[i] = D_val * u_val;
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            constexpr float kLog2e = M_LOG2E;
            weight_t A_val = A[state_idx * params.A_dstate_stride];
            A_val *= kLog2e;
            weight_t B_vals[kNItems], C_vals[kNItems];
            load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));
            load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight1, (params.seqlen - chunk * kChunkSize));
            __syncthreads();
            scan_t thread_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                thread_data[i] = make_float2(exp2f(delta_vals[i] * A_val), B_vals[i] * delta_u_vals[i]);
                if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                    if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                        thread_data[i] = make_float2(1.f, 0.f);
                    }
                }
            }
            // Initialize running total
            scan_t running_prefix;
            // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
            running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
            // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
            );
            // There's a syncthreads in the scan op, so we don't need to sync here.
            // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
            if (threadIdx.x == 0) {
                smem_running_prefix[state_idx] = prefix_op.running_prefix;
                x[chunk * params.dstate + state_idx] = prefix_op.running_prefix;
            }
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                out_vals[i] += thread_data[i].y * C_vals[i];
            }
        }

        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(out, out_vals, smem_store, params.seqlen - chunk * kChunkSize);
        Bvar += kChunkSize;
        Cvar += kChunkSize;
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, input_t, weight_t>;
        constexpr int kSmemSize = Ktraits::kSmemSize + Ktraits::MaxDState * sizeof(typename Ktraits::scan_t);
        // printf("smem_size = %d\n", kSmemSize);
        dim3 grid(params.batch, params.dim);
        auto kernel = &selective_scan_fwd_kernel<Ktraits>;
        if (kSmemSize >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<int knrows, typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    } else {
        selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
}
