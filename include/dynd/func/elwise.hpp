#include <dynd/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    [[deprecated("Using elwise from the header <dynd/func/elwise.hpp> is deprecated. Please stop using that header. "
                 "dynd::nd::functional::elwise "
                 "is now provided in <dynd/functional.hpp>.")]] DYND_API callable
    elwise(const callable &child);
  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
