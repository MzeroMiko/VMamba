/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "reverse_scan.cuh"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_, bool kDeltaSoftplus_, typename input_t_, typename weight_t_>
struct Selective_Scan_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int MaxDState = MAX_DSTATE / kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    static constexpr int kMinBlocks = kNThreads == 128 && 3;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItems>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 2 * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = 2 * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bwd_kernel(SSMParamsBwd params) {
    constexpr bool kDeltaSoftplus = Ktraits::kDeltaSoftplus;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    weight_t *smem_delta_a = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize);
    // scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + kNRows * (2 * Ktraits::MaxDState + kNThreads));
    scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + kNRows * 2 * Ktraits::MaxDState + kNThreads);
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_running_postfix + kNRows * Ktraits::MaxDState);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int dim_id_nrow = dim_id * kNRows;
    const int group_id = dim_id_nrow / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id_nrow * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id_nrow * params.delta_d_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id_nrow * params.dout_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id_nrow * params.A_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id_nrow * params.dA_d_stride;
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (batch_id * params.dB_batch_stride + group_id * params.dB_group_stride);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (batch_id * params.dC_batch_stride + group_id * params.dC_group_stride);
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id_nrow;
    float *D_val = params.D_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.D_ptr) + dim_id_nrow;
    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id_nrow;
    float *delta_bias = params.delta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.delta_bias_ptr) + dim_id_nrow;
    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id_nrow) * (params.n_chunks) * params.dstate;
    float dD_val[kNRows] = {0};
    float ddelta_bias_val[kNRows] = {0};

    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize;
    delta += (params.n_chunks - 1) * kChunkSize;
    dout += (params.n_chunks - 1) * kChunkSize;
    Bvar += (params.n_chunks - 1) * kChunkSize;
    Cvar += (params.n_chunks - 1) * kChunkSize;
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        input_t u_vals[kNRows][kNItems];
        input_t delta_vals_load[kNRows][kNItems];
        input_t dout_vals_load[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            __syncthreads();
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            __syncthreads();
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            __syncthreads();
            load_input<Ktraits>(dout + r * params.dout_d_stride, dout_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u -= kChunkSize;
        // Will reload delta at the same location if kDeltaSoftplus
        if constexpr (!kDeltaSoftplus) { delta -= kChunkSize; }
        dout -= kChunkSize;

        float dout_vals[kNRows][kNItems], delta_vals[kNRows][kNItems];
        float du_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                dout_vals[r][i] = float(dout_vals_load[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + (delta_bias == nullptr? 0: delta_bias[r]);
                if constexpr (kDeltaSoftplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
            }
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) { du_vals[r][i] = (D_val == nullptr? 0: D_val[r]) * dout_vals[r][i]; }
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) { dD_val[r] += dout_vals[r][i] * float(u_vals[r][i]); }
        }

        float ddelta_vals[kNRows][kNItems] = {0};
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            weight_t A_scaled[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                constexpr float kLog2e = M_LOG2E;
                A_scaled[r] = A_val[r] * kLog2e;
            }
            weight_t B_vals[kNItems], C_vals[kNItems];
            load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));
            auto &smem_load_weight_C = smem_load_weight1;
            load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize));
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                scan_t thread_data[kNItems], thread_reverse_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const float delta_a_exp = exp2f(delta_vals[r][i] * A_scaled[r]);
                    thread_data[i] = make_float2(delta_a_exp, delta_vals[r][i] * float(u_vals[r][i]) * B_vals[i]);
                    if (i == 0) {
                        // smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * Ktraits::MaxDState + r * (2 * Ktraits::MaxDState + kNThreads) : threadIdx.x + 2 * Ktraits::MaxDState + r * (2 * Ktraits::MaxDState + kNThreads)] = delta_a_exp;
                        smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * Ktraits::MaxDState + r * 2 * Ktraits::MaxDState : threadIdx.x + kNRows * 2 * Ktraits::MaxDState] = delta_a_exp;
                        
                    } else {
                        thread_reverse_data[i - 1].x = delta_a_exp;
                    }
                    thread_reverse_data[i].y = dout_vals[r][i] * C_vals[i];
                }
                __syncthreads();
                thread_reverse_data[kNItems - 1].x = threadIdx.x == kNThreads - 1
                    // ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * Ktraits::MaxDState + r * (2 * Ktraits::MaxDState + kNThreads)])
                    // : smem_delta_a[threadIdx.x + 1 + 2 * Ktraits::MaxDState + r * (2 * Ktraits::MaxDState + kNThreads)];
                    ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * Ktraits::MaxDState + r * 2 * Ktraits::MaxDState])
                    : smem_delta_a[threadIdx.x + 1 + kNRows * 2 * Ktraits::MaxDState];
                // Initialize running total
                scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(r * params.n_chunks + chunk - 1) * params.dstate + state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx + r * Ktraits::MaxDState] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                    thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                );
                if (threadIdx.x == 0) { smem_running_postfix[state_idx + r * Ktraits::MaxDState] = postfix_op.running_prefix; }
                weight_t dA_val = 0;
                weight_t dB_vals[kNItems], dC_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const float dx = thread_reverse_data[i].y;
                    const float ddelta_u = dx * B_vals[i];
                    du_vals[r][i] += ddelta_u * delta_vals[r][i];
                    const float a = thread_data[i].y - (delta_vals[r][i] * float(u_vals[r][i]) * B_vals[i]);
                    ddelta_vals[r][i] += ddelta_u * float(u_vals[r][i]) + dx * A_val[r] * a;
                    dA_val += dx * delta_vals[r][i] * a;
                    dB_vals[i] = dx * delta_vals[r][i] * float(u_vals[r][i]);
                    dC_vals[i] = dout_vals[r][i] * thread_data[i].y;
                }
                // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                    Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
                    auto &smem_exchange_C = smem_exchange1;
                    Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
                    const int seqlen_remaining = params.seqlen - chunk * kChunkSize - threadIdx.x;
                    weight_t *dB_cur = dB + state_idx * params.dB_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    weight_t *dC_cur = dC + state_idx * params.dC_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                            { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                        }
                    }
                dA_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dA_val);
                if (threadIdx.x == 0) {
                    smem_da[state_idx + r * Ktraits::MaxDState] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx + r * Ktraits::MaxDState];
                }
            }
        }

        if constexpr (kDeltaSoftplus) {
            input_t delta_vals_load[kNRows][kNItems];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                __syncthreads();
                load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            }
            delta -= kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float delta_val = float(delta_vals_load[r][i]) + (delta_bias == nullptr? 0: delta_bias[r]);
                    float delta_val_neg_exp = expf(-delta_val);
                    ddelta_vals[r][i] = delta_val <= 20.f
                        ? ddelta_vals[r][i] / (1.f + delta_val_neg_exp)
                        : ddelta_vals[r][i];
                }
            }
        }
        
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) { ddelta_bias_val[r] += ddelta_vals[r][i]; }
        }

        input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride
            + dim_id_nrow * params.du_d_stride + chunk * kChunkSize;
        input_t *ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
            + dim_id_nrow * params.ddelta_d_stride + chunk * kChunkSize;
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            __syncthreads();
            store_output<Ktraits>(du + r * params.du_d_stride, du_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            __syncthreads();
            store_output<Ktraits>(ddelta + r * params.ddelta_d_stride, ddelta_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        Bvar -= kChunkSize;
        Cvar -= kChunkSize;
    }

    #pragma unroll
    for (int r = 0; r < kNRows; ++r) {
        if (params.dD_ptr != nullptr) {
            __syncthreads();
            dD_val[r] = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val[r]);
            if (threadIdx.x == 0) { gpuAtomicAdd(&(dD[r]), dD_val[r]); }
        }
        if (params.ddelta_bias_ptr != nullptr) {
            __syncthreads();
            ddelta_bias_val[r] = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddelta_bias_val[r]);
            if (threadIdx.x == 0) { gpuAtomicAdd(&(ddelta_bias[r]), ddelta_bias_val[r]); }
        }
        __syncthreads();
        for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
            gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride + r * params.dA_d_stride]), smem_da[state_idx + r * Ktraits::MaxDState]);
        }
    }
}

template<int kNThreads, int kNItems, int kNRows, typename input_t, typename weight_t>
void selective_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.delta_softplus, kDeltaSoftplus, [&] {
            using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kDeltaSoftplus, input_t, weight_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * Ktraits::MaxDState * sizeof(typename Ktraits::scan_t) + (kNThreads + 4 * kNRows * Ktraits::MaxDState) * sizeof(typename Ktraits::weight_t);
            // printf("smem_size = %d\n", kSmemSize);
            dim3 grid(params.batch, params.dim / kNRows);
            auto kernel = &selective_scan_bwd_kernel<Ktraits>;
            if (kSmemSize >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            }
            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template<int knrows, typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_bwd_launch<32, 4, knrows, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_bwd_launch<32, 8, knrows, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_bwd_launch<32, 16, knrows, input_t, weight_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_bwd_launch<64, 16, knrows, input_t, weight_t>(params, stream);
    } else {
        selective_scan_bwd_launch<128, 16, knrows, input_t, weight_t>(params, stream);
    }
}