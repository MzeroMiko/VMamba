#include <stdio.h>
// On pytorch 1.10 and CUDA 10.2, I get compilation errors on torch/csrc/api/include/torch/nn/cloneable.h
// So we'll only include torch/python.h instead of torch/extension.h
// Similar to https://github.com/getkeops/keops/blob/3efd428b55c724b12f23982c06de00bc4d02d903/pykeops/torch_headers.h.in#L8
// #include <torch/extension.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>  // For getCurrentCUDAStream
#include <THC/THCAtomics.cuh>  // For atomicAdd on complex
#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/util/complex.h>  // For scalar_value_type
#include "map.h"  // For the MAP macro, i.e. for_each over the arguments


#ifndef ITEMS_PER_THREAD_SYM_FWD_VALUES
  #define ITEMS_PER_THREAD_SYM_FWD_VALUES {2, 4, 8, 16, 32, 32, 32, 64, 64, 64}
#endif
#ifndef MAX_BLOCK_SIZE_VALUE
  #define MAX_BLOCK_SIZE_VALUE 256
#endif
#ifndef ITEMS_PER_THREAD_SYM_BWD_VALUE
  #define ITEMS_PER_THREAD_SYM_BWD_VALUE 32
#endif

static constexpr int ITEMS_PER_THREAD_SYM_FWD[] = ITEMS_PER_THREAD_SYM_FWD_VALUES;
static constexpr int MAX_BLOCK_SIZE = MAX_BLOCK_SIZE_VALUE;
static constexpr int ITEMS_PER_THREAD_SYM_BWD = ITEMS_PER_THREAD_SYM_BWD_VALUE;

template <typename T, size_t N>
using CudaAcsr = at::GenericPackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;
constexpr __host__ __device__ int div_up_const(int a, int b) { return (a + b - 1) / b; }

__host__ __device__ static inline int div_up(int a, int b) { return (a + b - 1) / b;}

template <typename scalar_t, int log_N,
            int items_per_thread=ITEMS_PER_THREAD_SYM_FWD[log_N - 1]>
__global__ void cauchy_mult_sym_fwd_cuda_kernel(CudaAcsr<scalar_t, 2> v,
                                                CudaAcsr<scalar_t, 1> z,
                                                CudaAcsr<scalar_t, 2> w,
                                                CudaAcsr<scalar_t, 2> out,
                                                int L) {
  // Get the float type from the complex type
  // https://github.com/pytorch/pytorch/blob/bceb1db885cafa87fe8d037d8f22ae9649a1bba0/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp#L213
  using float_t = typename at::scalar_value_type<scalar_t>::type;
  constexpr int N = 1 << log_N;
  constexpr int blockDimx = div_up_const(N, items_per_thread);
  constexpr int blockDimy = MAX_BLOCK_SIZE / blockDimx;
  // We just want a shared array:
  // __shared__ scalar_t s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  __shared__ char v_smem_char[N * sizeof(scalar_t)];
  scalar_t *v_smem = (scalar_t *)&v_smem_char;
  __shared__ char w_smem_char[N * sizeof(scalar_t)];
  scalar_t *w_smem = (scalar_t *)&w_smem_char;
  __shared__ char out_smem_char[blockDimy * sizeof(scalar_t)];
  scalar_t *out_smem = (scalar_t *)&out_smem_char;
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int L_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int L_block_start = blockIdx.y * blockDim.y;
  scalar_t z_t = L_block_start + threadIdx.y < L ? z[L_block_start + threadIdx.y] : scalar_t(0.f);
  for (int N_idx = threadIdx.x + threadIdx.y * blockDim.x; N_idx < N; N_idx += blockDim.x * blockDim.y) {
    v_smem[N_idx] = v[batch_idx][N_idx];
    w_smem[N_idx] = w[batch_idx][N_idx];
  }
  __syncthreads();
  scalar_t result = 0;
  if (L_idx < L) {
    // Combining the two terms (a/b + c/d = (ad + bc)/(bd)) seems to increase numerical errors.
    // Using nvcc --use_fast_math yields the same speed between the two versions.
    // So we don't combine the two terms.
    #pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
      int N_idx = item * blockDimx + threadIdx.x;
      scalar_t v_t = v_smem[N_idx], w_t = w_smem[N_idx];
      result += v_t / (z_t - w_t) + std::conj(v_t) / (z_t - std::conj(w_t));
    }
  }
  // TODO: this only works for N a power of 2
  #pragma unroll
  for (int offset = blockDimx / 2; offset > 0; offset /= 2) {
    result += WARP_SHFL_DOWN(result, offset);
  }
  if ((threadIdx.x == 0) && (L_idx < L)) {
    out_smem[threadIdx.y] = result;
  }
  __syncthreads();
  if (tid < blockDim.y && L_block_start + tid < L) {
    out[batch_idx][L_block_start + tid] = out_smem[tid];
  }
}

torch::Tensor cauchy_mult_sym_fwd_cuda(torch::Tensor v,
                                       torch::Tensor z,
                                       torch::Tensor w) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  const int L = z.size(0);
  auto out = torch::empty({batch_size, L}, torch::dtype(v.dtype()).device(v.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<float>;
  const auto v_a = v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto z_a = z.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
  const auto w_a = w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto out_a = out.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  int log_N = int(log2((double) N));
  int block_x = div_up(N, ITEMS_PER_THREAD_SYM_FWD[log_N - 1]);
  dim3 block(block_x, MAX_BLOCK_SIZE / block_x);
  dim3 grid(batch_size, div_up(L, block.y));
  switch (log_N) {
    #define CASE_LOG_N(log_N_val) case log_N_val:                 \
    cauchy_mult_sym_fwd_cuda_kernel<scalar_t, log_N_val>          \
      <<<grid, block, 0, stream>>>(v_a, z_a, w_a, out_a, L); break;
    MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  }
  #undef CASE_LOG_N
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <typename scalar_t, bool check_L_boundary>
__global__ void cauchy_mult_sym_bwd_cuda_kernel(CudaAcsr<scalar_t, 2> v,
                                                CudaAcsr<scalar_t, 1> z,
                                                CudaAcsr<scalar_t, 2> w,
                                                CudaAcsr<scalar_t, 2> dout,
                                                CudaAcsr<scalar_t, 3> dv,
                                                CudaAcsr<scalar_t, 3> dw,
                                                int L,
                                                int L_chunk_size) {
  // We just want a shared array:
  // __shared__ scalar_t s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  __shared__ char dv_smem_char[C10_WARP_SIZE * sizeof(scalar_t)];
  scalar_t *dv_smem = (scalar_t *)&dv_smem_char;
  __shared__ char dw_smem_char[C10_WARP_SIZE * sizeof(scalar_t)];
  scalar_t *dw_smem = (scalar_t *)&dw_smem_char;
  int batch_idx = blockIdx.x;
  int N_idx = blockIdx.y;
  int L_chunk_idx = blockIdx.z;
  int tid = threadIdx.x;
  scalar_t w_conj_t = std::conj(w[batch_idx][N_idx]);
  scalar_t dv_t = 0;
  scalar_t dw_t = 0;
  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD_SYM_BWD; ++item) {
    int l = L_chunk_idx * L_chunk_size + item * blockDim.x + threadIdx.x;
    scalar_t dout_t, z_t;
    if (check_L_boundary) {
      dout_t = l < L ? dout[batch_idx][l] : 0;
      z_t = l < L ? z[l] : 1;
    } else { // Not checking boundary can speed it up quite a bit, around 30%.
      dout_t = dout[batch_idx][l];
      z_t = z[l];
    }
    scalar_t denom_1 = std::conj(z_t) - w_conj_t;
    scalar_t denom_2 = z_t - w_conj_t;
    scalar_t term_1 = dout_t / denom_1;
    scalar_t term_2 = std::conj(dout_t) / denom_2;
    dv_t += term_1 + term_2;
    dw_t += term_1 / denom_1 + term_2 / denom_2;
  }
  dv_t = at::native::cuda_utils::BlockReduceSum<scalar_t>(dv_t, dv_smem);
  dw_t = at::native::cuda_utils::BlockReduceSum<scalar_t>(dw_t, dw_smem);
  if (tid == 0) {
    dw[batch_idx][N_idx][L_chunk_idx] = dw_t * std::conj(v[batch_idx][N_idx]);
    dv[batch_idx][N_idx][L_chunk_idx] = dv_t;
  }
}

std::tuple<torch::Tensor, torch::Tensor>
cauchy_mult_sym_bwd_cuda(torch::Tensor v,
                         torch::Tensor z,
                         torch::Tensor w,
                         torch::Tensor dout) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  const int L = z.size(0);
  constexpr int MAX_BLOCK_SIZE = 1024;
  constexpr int MAX_L_CHUNK_SIZE = ITEMS_PER_THREAD_SYM_BWD * MAX_BLOCK_SIZE;
  const int n_L_chunks = div_up(L, MAX_L_CHUNK_SIZE);
  auto dv = torch::empty({batch_size, N, n_L_chunks}, torch::dtype(v.dtype()).device(v.device()));
  auto dw = torch::empty({batch_size, N, n_L_chunks}, torch::dtype(w.dtype()).device(w.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  using scalar_t = c10::complex<float>;
  const auto v_a = v.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto z_a = z.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
  const auto w_a = w.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  const auto dout_a = dout.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
  auto dv_a = dv.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
  auto dw_a = dw.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
  // Each block need to have a multiple of 32 threads, otherwise
  // at::native::cuda_utils::BlockReduceSum to produce wrong result.
  // int block_x = max(div_up(L, ITEMS_PER_THREAD_SYM_BWD), C10_WARP_SIZE);
  const int L_chunk_size = min(L, MAX_L_CHUNK_SIZE);
  int block_x = div_up(L_chunk_size, ITEMS_PER_THREAD_SYM_BWD * C10_WARP_SIZE) * C10_WARP_SIZE;
  bool check_L_boundary = L != block_x * ITEMS_PER_THREAD_SYM_BWD * n_L_chunks;
  dim3 block(block_x);
  dim3 grid(batch_size, N, n_L_chunks);
  check_L_boundary
    ? cauchy_mult_sym_bwd_cuda_kernel<scalar_t, true>
      <<<grid, block, 0, stream>>>(v_a, z_a, w_a, dout_a, dv_a, dw_a, L, L_chunk_size)
    : cauchy_mult_sym_bwd_cuda_kernel<scalar_t, false>
      <<<grid, block, 0, stream>>>(v_a, z_a, w_a, dout_a, dv_a, dw_a, L, L_chunk_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dv.sum(-1), dw.sum(-1));
}

template <int log_N, int items_per_thread=ITEMS_PER_THREAD_SYM_FWD[log_N - 1]>
__global__ void vand_log_mult_sym_fwd_cuda_kernel(CudaAcsr<c10::complex<float>, 2> v,
                                                  CudaAcsr<c10::complex<float>, 2> x,
                                                  CudaAcsr<float, 2> out,
                                                  int L) {
  using cfloat_t = typename c10::complex<float>;
  constexpr int N = 1 << log_N;
  constexpr int blockDimx = div_up_const(N, items_per_thread);
  constexpr int blockDimy = MAX_BLOCK_SIZE / blockDimx;
  // We just want a shared array:
  // __shared__ cfloat_t s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  __shared__ char v_smem_char[N * sizeof(cfloat_t)];
  cfloat_t *v_smem = (cfloat_t *)&v_smem_char;
  __shared__ char x_smem_char[N * sizeof(cfloat_t)];
  cfloat_t *x_smem = (cfloat_t *)&x_smem_char;
  __shared__ float out_smem[blockDimy];
  int batch_idx = blockIdx.x;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int L_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int L_block_start = blockIdx.y * blockDim.y;
  for (int N_idx = threadIdx.x + threadIdx.y * blockDim.x; N_idx < N; N_idx += blockDim.x * blockDim.y) {
    v_smem[N_idx] = v[batch_idx][N_idx];
    x_smem[N_idx] = x[batch_idx][N_idx];
  }
  __syncthreads();
  float result = 0;
  if (L_idx < L) {
    #pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
      int N_idx = item * blockDimx + threadIdx.x;
      cfloat_t v_t = v_smem[N_idx], x_t = x_smem[N_idx];
      result += (std::exp(x_t * L_idx) * v_t).real_;
    }
  }
  // TODO: this only works for N a power of 2
  #pragma unroll
  for (int offset = blockDimx / 2; offset > 0; offset /= 2) {
    result += WARP_SHFL_DOWN(result, offset);
  }
  if ((threadIdx.x == 0) && (L_idx < L)) {
    out_smem[threadIdx.y] = 2 * result;
  }
  __syncthreads();
  if (tid < blockDim.y && L_block_start + tid < L) {
    out[batch_idx][L_block_start + tid] = out_smem[tid];
  }
}

torch::Tensor vand_log_mult_sym_fwd_cuda(torch::Tensor v, torch::Tensor x, int L) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  auto opts = v.options();
  auto out = torch::empty({batch_size, L}, opts.dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto v_a = v.packed_accessor32<c10::complex<float>, 2, at::RestrictPtrTraits>();
  const auto x_a = x.packed_accessor32<c10::complex<float>, 2, at::RestrictPtrTraits>();
  auto out_a = out.packed_accessor32<float, 2, at::RestrictPtrTraits>();
  int log_N = int(log2((double) N));
  int block_x = div_up(N, ITEMS_PER_THREAD_SYM_FWD[log_N - 1]);
  dim3 block(block_x, MAX_BLOCK_SIZE / block_x);
  dim3 grid(batch_size, div_up(L, block.y));
  switch (log_N) {
    #define CASE_LOG_N(log_N_val) case log_N_val:                 \
    vand_log_mult_sym_fwd_cuda_kernel<log_N_val>          \
      <<<grid, block, 0, stream>>>(v_a, x_a, out_a, L); break;
    MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  }
  #undef CASE_LOG_N
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <bool check_L_boundary>
__global__ void vand_log_mult_sym_bwd_cuda_kernel(CudaAcsr<c10::complex<float>, 2> v,
                                                  CudaAcsr<c10::complex<float>, 2> x,
                                                  CudaAcsr<float, 2> dout,
                                                  CudaAcsr<c10::complex<float>, 3> dv,
                                                  CudaAcsr<c10::complex<float>, 3> dx,
                                                  int L,
                                                  int L_chunk_size) {
  using cfloat_t = typename c10::complex<float>;
  // We just want a shared array:
  // __shared__ c10::complex<float> s_b[16];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  __shared__ char dv_smem_char[C10_WARP_SIZE * sizeof(cfloat_t)];
  cfloat_t *dv_smem = (cfloat_t *)&dv_smem_char;
  __shared__ char dx_smem_char[C10_WARP_SIZE * sizeof(cfloat_t)];
  cfloat_t *dx_smem = (cfloat_t *)&dx_smem_char;
  int batch_idx = blockIdx.x;
  int N_idx = blockIdx.y;
  int L_chunk_idx = blockIdx.z;
  int tid = threadIdx.x;
  cfloat_t x_t = x[batch_idx][N_idx];
  cfloat_t dv_t = 0;
  cfloat_t dx_t = 0;
  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD_SYM_BWD; ++item) {
    int l = L_chunk_idx * L_chunk_size + item * blockDim.x + threadIdx.x;
    float dout_t;
    if (check_L_boundary) {
      dout_t = l < L ? dout[batch_idx][l] : 0;
    } else { // Not checking boundary can speed it up quite a bit.
      dout_t = dout[batch_idx][l];
    }
    // Need to conjugate as we're doing complex gradient.
    cfloat_t do_exp_x_t = dout_t * std::conj(std::exp(x_t * l));
    dv_t += do_exp_x_t;
    dx_t += do_exp_x_t * l;
  }
  dv_t = at::native::cuda_utils::BlockReduceSum<cfloat_t>(dv_t, dv_smem);
  dx_t = at::native::cuda_utils::BlockReduceSum<cfloat_t>(dx_t, dx_smem);
  if (tid == 0) {
    dx[batch_idx][N_idx][L_chunk_idx] = 2 * dx_t * std::conj(v[batch_idx][N_idx]);
    dv[batch_idx][N_idx][L_chunk_idx] = 2 * dv_t;
  }
}


std::tuple<torch::Tensor, torch::Tensor>
vand_log_mult_sym_bwd_cuda(torch::Tensor v,
                           torch::Tensor x,
                           torch::Tensor dout) {
  const int batch_size = v.size(0);
  const int N = v.size(1);
  const int L = dout.size(1);
  constexpr int MAX_BLOCK_SIZE = 1024;
  constexpr int MAX_L_CHUNK_SIZE = ITEMS_PER_THREAD_SYM_BWD * MAX_BLOCK_SIZE;
  const int n_L_chunks = div_up(L, MAX_L_CHUNK_SIZE);
  auto dv = torch::empty({batch_size, N, n_L_chunks}, torch::dtype(v.dtype()).device(v.device()));
  auto dx = torch::empty({batch_size, N, n_L_chunks}, torch::dtype(x.dtype()).device(x.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  using cfloat_t = c10::complex<float>;
  const auto v_a = v.packed_accessor32<cfloat_t, 2, at::RestrictPtrTraits>();
  const auto x_a = x.packed_accessor32<cfloat_t, 2, at::RestrictPtrTraits>();
  const auto dout_a = dout.packed_accessor32<float, 2, at::RestrictPtrTraits>();
  auto dv_a = dv.packed_accessor32<cfloat_t, 3, at::RestrictPtrTraits>();
  auto dx_a = dx.packed_accessor32<cfloat_t, 3, at::RestrictPtrTraits>();
  // Each block need to have a multiple of 32 threads, otherwise
  // at::native::cuda_utils::BlockReduceSum to produce wrong result.
  // int block_x = max(div_up(L, ITEMS_PER_THREAD_SYM_BWD), C10_WARP_SIZE);
  const int L_chunk_size = min(L, MAX_L_CHUNK_SIZE);
  int block_x = div_up(L_chunk_size, ITEMS_PER_THREAD_SYM_BWD * C10_WARP_SIZE) * C10_WARP_SIZE;
  bool check_L_boundary = L != block_x * ITEMS_PER_THREAD_SYM_BWD * n_L_chunks;
  dim3 block(block_x);
  dim3 grid(batch_size, N, n_L_chunks);
  check_L_boundary
    ? vand_log_mult_sym_bwd_cuda_kernel<true>
      <<<grid, block, 0, stream>>>(v_a, x_a, dout_a, dv_a, dx_a, L, L_chunk_size)
    : vand_log_mult_sym_bwd_cuda_kernel<false>
      <<<grid, block, 0, stream>>>(v_a, x_a, dout_a, dv_a, dx_a, L, L_chunk_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dv.sum(-1), dx.sum(-1));
}
