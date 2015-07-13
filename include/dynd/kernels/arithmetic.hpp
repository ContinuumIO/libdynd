#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t I0>
  struct plus_kernel
      : base_kernel<plus_kernel<I0>, kernel_request_cuda_host_device, 1> {
    typedef typename type_of<I0>::type A0;
    typedef decltype(+std::declval<A0>()) dst_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) = +*reinterpret_cast<A0 *>(src[0]);
    }
  };

  template <type_id_t I0>
  struct minus_kernel
      : base_kernel<minus_kernel<I0>, kernel_request_cuda_host_device, 1> {
    typedef typename type_of<I0>::type A0;
    typedef decltype(-std::declval<A0>()) dst_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) = -*reinterpret_cast<A0 *>(src[0]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct add_kernel
      : base_kernel<add_kernel<I0, I1>, kernel_request_cuda_host_device, 2> {
    typedef add_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() + std::declval<A1>()) dst_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) + *reinterpret_cast<A1 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct subtract_kernel : base_kernel<subtract_kernel<I0, I1>,
                                       kernel_request_cuda_host_device, 2> {
    typedef subtract_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() - std::declval<A1>()) dst_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) - *reinterpret_cast<A1 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct multiply_kernel : base_kernel<multiply_kernel<I0, I1>,
                                       kernel_request_cuda_host_device, 2> {
    typedef multiply_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() * std::declval<A1>()) dst_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) * *reinterpret_cast<A1 *>(src[1]);
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct divide_kernel
      : base_kernel<divide_kernel<I0, I1>, kernel_request_cuda_host_device, 2> {
    typedef divide_kernel self_type;
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef decltype(std::declval<A0>() / std::declval<A1>()) dst_type;

    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      *reinterpret_cast<dst_type *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) / *reinterpret_cast<A1 *>(src[1]);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID>
  struct type::equivalent<nd::plus_kernel<Src0TypeID>> {
    static type make()
    {
      typedef nd::plus_kernel<Src0TypeID> kernel_type;
      return arrfunc_type::make(type::make<typename kernel_type::dst_type>(),
                                {type::make<Src0TypeID>()});
    }
  };

  template <type_id_t Src0TypeID>
  struct type::equivalent<nd::minus_kernel<Src0TypeID>> {
    static type make()
    {
      typedef nd::minus_kernel<Src0TypeID> kernel_type;
      return arrfunc_type::make(type::make<typename kernel_type::dst_type>(),
                                {type::make<Src0TypeID>()});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::add_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      typedef nd::add_kernel<Src0TypeID, Src1TypeID> kernel_type;
      return arrfunc_type::make(
          type::make<typename kernel_type::dst_type>(),
          {type::make<Src0TypeID>(), type::make<Src1TypeID>()});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::subtract_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      typedef nd::subtract_kernel<Src0TypeID, Src1TypeID> kernel_type;
      return arrfunc_type::make(
          type::make<typename kernel_type::dst_type>(),
          {type::make<Src0TypeID>(), type::make<Src1TypeID>()});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::multiply_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      typedef nd::multiply_kernel<Src0TypeID, Src1TypeID> kernel_type;
      return arrfunc_type::make(
          type::make<typename kernel_type::dst_type>(),
          {type::make<Src0TypeID>(), type::make<Src1TypeID>()});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::divide_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      typedef nd::divide_kernel<Src0TypeID, Src1TypeID> kernel_type;
      return arrfunc_type::make(
          type::make<typename kernel_type::dst_type>(),
          {type::make<Src0TypeID>(), type::make<Src1TypeID>()});
    }
  };

} // namespace dynd::ndt
} // namespace dynd