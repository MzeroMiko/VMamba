#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define PTR(x) ((void *__restrict__)(x.data_ptr()))

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

template <typename input_t>
void cross_scan_cuda(void *__restrict__  x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);

template <typename input_t>
void cross_merge_cuda(void *__restrict__  x, void *__restrict__ out, int B, int C, int H, int W, cudaStream_t stream);

at::Tensor cross_scan_fwd(at::Tensor & xs) {
    const auto sizes = xs.sizes();
    const int B = sizes[0];
    const int C = sizes[1];
    const int H = sizes[2];
    const int W = sizes[3];
    CHECK_INPUT(xs);
    at::Tensor out = torch::empty({B, 4, C, H, W}, xs.options().device(torch::kCUDA));
    at::cuda::CUDAGuard device_guard{(char)xs.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(xs.scalar_type(), "cross_scan", [&] {
        cross_scan_cuda<input_t>(PTR(xs), PTR(out), B, C, H, W, stream);
    });
    return out;
}

at::Tensor cross_merge_fwd(at::Tensor & xs) {
    const auto sizes = xs.sizes();
    const int B = sizes[0];
    const int C = sizes[2];
    const int H = sizes[3];
    const int W = sizes[4];
    CHECK_INPUT(xs);
    at::Tensor out = torch::zeros({B, C, H, W}, xs.options().device(torch::kCUDA));
    at::cuda::CUDAGuard device_guard{(char)xs.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(xs.scalar_type(), "cross_merge", [&] {
        cross_merge_cuda<input_t>(PTR(xs), PTR(out), B, C, H, W, stream);
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_scan", &cross_scan_fwd, "cross_scan");
    m.def("cross_merge", &cross_merge_fwd, "cross_merge");
}
