//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/conj_callable.hpp>
#include <dynd/callables/imag_callable.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/callables/real_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/math.hpp>

using namespace std;
using namespace dynd;

namespace {
// CUDA and MSVC 2015 WORKAROUND: Using these functions directly in the apply
//                                template does not compile.
double mycos(double x) { return cos(x); }
double mysin(double x) { return sin(x); }
double mytan(double x) { return tan(x); }
double myexp(double x) { return exp(x); }

} // anonymous namespace

DYND_API nd::callable nd::cos = nd::functional::elwise(nd::functional::apply<double (*)(double), &mycos>());
DYND_API nd::callable nd::sin = nd::functional::elwise(nd::functional::apply<double (*)(double), &mysin>());
DYND_API nd::callable nd::tan = nd::functional::elwise(nd::functional::apply<double (*)(double), &mytan>());
DYND_API nd::callable nd::exp = nd::functional::elwise(nd::functional::apply<double (*)(double), &myexp>());

DYND_API nd::callable nd::real = nd::functional::elwise(nd::make_callable<nd::multidispatch_callable<1>>(
    ndt::type("(Scalar) -> Scalar"),
    nd::callable::make_all<nd::real_callable, type_sequence<dynd::complex<float>, dynd::complex<double>>>(
        [](const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
           const ndt::type *src_tp) -> std::vector<ndt::type> { return {src_tp[0]}; })));

DYND_API nd::callable nd::imag = nd::functional::elwise(nd::make_callable<nd::multidispatch_callable<1>>(
    ndt::type("(Scalar) -> Scalar"),
    nd::callable::make_all<nd::imag_callable, type_sequence<dynd::complex<float>, dynd::complex<double>>>(
        [](const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
           const ndt::type *src_tp) -> std::vector<ndt::type> { return {src_tp[0]}; })));

DYND_API nd::callable nd::conj = nd::functional::elwise(nd::make_callable<nd::multidispatch_callable<1>>(
    ndt::type("(Scalar) -> Scalar"),
    nd::callable::make_all<nd::conj_callable, type_sequence<dynd::complex<float>, dynd::complex<double>>>(
        [](const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
           const ndt::type *src_tp) -> std::vector<ndt::type> { return {src_tp[0]}; })));
