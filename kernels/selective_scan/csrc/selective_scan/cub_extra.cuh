// WarpMask is copied from /usr/local/cuda-12.1/include/cub/util_ptx.cuh
// PowerOfTwo is copied from /usr/local/cuda-12.1/include/cub/util_type.cuh

#pragma once

#include <cub/util_type.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_debug.cuh>

/**
 * \brief Statically determine if N is a power-of-two
 */
 template <int N>
 struct PowerOfTwo
 {
     enum { VALUE = ((N & (N - 1)) == 0) };
 };
 

/**
 * @brief Returns the warp mask for a warp of @p LOGICAL_WARP_THREADS threads
 *
 * @par
 * If the number of threads assigned to the virtual warp is not a power of two,
 * it's assumed that only one virtual warp exists.
 *
 * @tparam LOGICAL_WARP_THREADS <b>[optional]</b> The number of threads per
 *                              "logical" warp (may be less than the number of
 *                              hardware warp threads).
 * @param warp_id Id of virtual warp within architectural warp
 */
 template <int LOGICAL_WARP_THREADS, int LEGACY_PTX_ARCH = 0>
 __host__ __device__ __forceinline__
 unsigned int WarpMask(unsigned int warp_id)
 {
   constexpr bool is_pow_of_two = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;
   constexpr bool is_arch_warp  = LOGICAL_WARP_THREADS == CUB_WARP_THREADS(0);
 
   unsigned int member_mask = 0xFFFFFFFFu >>
                              (CUB_WARP_THREADS(0) - LOGICAL_WARP_THREADS);
 
   if (is_pow_of_two && !is_arch_warp)
   {
     member_mask <<= warp_id * LOGICAL_WARP_THREADS;
   }
 
   return member_mask;
 }
 