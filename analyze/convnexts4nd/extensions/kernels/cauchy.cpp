#include <vector>
#include <utility>
#include <cmath>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

torch::Tensor cauchy_mult_sym_fwd_cuda(torch::Tensor v,
                                       torch::Tensor z,
                                       torch::Tensor w);
std::tuple<torch::Tensor, torch::Tensor> cauchy_mult_sym_bwd_cuda(torch::Tensor v,
                                                                  torch::Tensor z,
                                                                  torch::Tensor w,
                                                                  torch::Tensor dout);

namespace cauchy {

torch::Tensor cauchy_mult_sym_fwd(torch::Tensor v,
                                  torch::Tensor z,
                                  torch::Tensor w) {
  CHECK_DEVICE(v); CHECK_DEVICE(z); CHECK_DEVICE(w);
  const auto batch_size = v.size(0);
  const auto N = v.size(1);
  const auto L = z.size(0);
  CHECK_SHAPE(v, batch_size, N);
  CHECK_SHAPE(z, L);
  CHECK_SHAPE(w, batch_size, N);
  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)v.get_device()};
  return cauchy_mult_sym_fwd_cuda(v, z, w);
}

std::tuple<torch::Tensor, torch::Tensor>
cauchy_mult_sym_bwd(torch::Tensor v,
                    torch::Tensor z,
                    torch::Tensor w,
                    torch::Tensor dout) {
  CHECK_DEVICE(v); CHECK_DEVICE(z); CHECK_DEVICE(w); CHECK_DEVICE(dout);
  const auto batch_size = v.size(0);
  const auto N = v.size(1);
  const auto L = z.size(0);
  CHECK_SHAPE(v, batch_size, N);
  CHECK_SHAPE(z, L);
  CHECK_SHAPE(w, batch_size, N);
  CHECK_SHAPE(dout, batch_size, L);
  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)v.get_device()};
  return cauchy_mult_sym_bwd_cuda(v, z, w, dout);
}

}  // cauchy

torch::Tensor vand_log_mult_sym_fwd_cuda(torch::Tensor v, torch::Tensor x, int L);

std::tuple<torch::Tensor, torch::Tensor>
vand_log_mult_sym_bwd_cuda(torch::Tensor v, torch::Tensor x, torch::Tensor dout);

namespace vand {

torch::Tensor vand_log_mult_sym_fwd(torch::Tensor v, torch::Tensor x, int L) {
  CHECK_DEVICE(v); CHECK_DEVICE(x);
  const auto batch_size = v.size(0);
  const auto N = v.size(1);
  CHECK_SHAPE(v, batch_size, N);
  CHECK_SHAPE(x, batch_size, N);
  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)v.get_device()};
  return vand_log_mult_sym_fwd_cuda(v, x, L);
}

std::tuple<torch::Tensor, torch::Tensor>
vand_log_mult_sym_bwd(torch::Tensor v, torch::Tensor x, torch::Tensor dout) {
  CHECK_DEVICE(v); CHECK_DEVICE(x); CHECK_DEVICE(dout);
  const auto batch_size = v.size(0);
  const auto N = v.size(1);
  const auto L = dout.size(1);
  CHECK_SHAPE(v, batch_size, N);
  CHECK_SHAPE(x, batch_size, N);
  CHECK_SHAPE(dout, batch_size, L);
  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)v.get_device()};
  return vand_log_mult_sym_bwd_cuda(v, x, dout);
}

}  // vand

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cauchy_mult_sym_fwd", &cauchy::cauchy_mult_sym_fwd,
        "Cauchy multiply symmetric forward");
  m.def("cauchy_mult_sym_bwd", &cauchy::cauchy_mult_sym_bwd,
        "Cauchy multiply symmetric backward");
  m.def("vand_log_mult_sym_fwd", &vand::vand_log_mult_sym_fwd,
        "Log Vandermonde multiply symmetric forward");
  m.def("vand_log_mult_sym_bwd", &vand::vand_log_mult_sym_bwd,
        "Log Vandermonde multiply symmetric backward");
}
