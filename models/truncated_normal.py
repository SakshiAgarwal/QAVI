import math
from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Normal
from torch.distributions.utils import _standard_normal, broadcast_all
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:'+str(1) if torch.cuda.is_available() else 'cpu')

'''
NOTICE this is derived from tensorflow_probability code. Original license text reproduced below.
'''

# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# The "ndtr" function is derived from calculations made in:
# https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
# In the following email exchange, the author gives his consent to redistribute
# derived works under an Apache 2.0 license.
#
# From: Stephen Moshier <steve@moshier.net>
# Date: Sat, Jun 9, 2018 at 2:36 PM
# Subject: Re: Licensing cephes under Apache (BSD-like) license.
# To: rif <rif@google.com>
#
#
#
# Hello Rif,
#
# Yes, Google may distribute Cephes files under the Apache 2 license.
#
# If clarification is needed, I do not favor BSD over other free licenses.
# I would agree that Apache 2 seems to cover the concern you mentioned
# about sublicensees.
#
# Best wishes for good luck with your projects!
# Steve Moshier
#
#
#
# On Thu, 31 May 2018, rif wrote:
#
# > Hello Steve.
# > My name is Rif. I work on machine learning software at Google.
# >
# > Your cephes software continues to be incredibly useful and widely used. I
# > was wondering whether it would be permissible for us to use the Cephes code
# > under the Apache 2.0 license, which is extremely similar in permissions to
# > the BSD license (Wikipedia comparisons). This would be quite helpful to us
# > in terms of avoiding multiple licenses on software.
# >
# > I'm sorry to bother you with this (I can imagine you're sick of hearing
# > about this by now), but I want to be absolutely clear we're on the level and
# > not misusing your important software. In former conversation with Eugene
# > Brevdo (ebrevdo@google.com), you wrote "If your licensing is similar to BSD,
# > the formal way that has been handled is simply to add a statement to the
# > effect that you are incorporating the Cephes software by permission of the
# > author." I wanted to confirm that (a) we could use the Apache license, (b)
# > that we don't need to (and probably you don't want to) keep getting
# > contacted about individual uses, because your intent is generally to allow
# > this software to be reused under "BSD-like" license, and (c) you're OK
# > letting incorporators decide whether a license is sufficiently BSD-like?
# >
# > Best,
# >
# > rif
# >
# >
# >

# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.) See `NdtrTest` in
# special_math_test.py.
LOGNDTR_FLOAT64_LOWER = -20.
LOGNDTR_FLOAT32_LOWER = -10.

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
LOGNDTR_FLOAT64_UPPER = 8.
LOGNDTR_FLOAT32_UPPER = 5.

def probit(x):
    return 0.5*(1+torch.erf(x/math.sqrt(2)))

def ndtr(x):
    """Normal distribution function.
    Returns the area under the Gaussian probability density function, integrated
    from minus infinity to x:
    ```
                      1       / x
       ndtr(x)  = ----------  |    exp(-0.5 t**2) dt
                  sqrt(2 pi)  /-inf
                = 0.5 (1 + erf(x / sqrt(2)))
                = 0.5 erfc(x / sqrt(2))
    ```
    Args:
      x: `Tensor` of type `float32`, `float64`.
      name: Python string. A name for the operation (default="ndtr").
    Returns:
      ndtr: `Tensor` with `dtype=x.dtype`.
    Raises:
      TypeError: if `x` is not floating-type.
    """
    return _ndtr(x)


def _ndtr(x):
    """Implements ndtr core logic."""
    half_sqrt_2 = 0.5 * np.sqrt(2.)
    w = x * half_sqrt_2
    z = torch.abs(w)
    y = torch.where(
        z < half_sqrt_2,
        1. + torch.erf(w),
        torch.where(w > 0., 2. - torch.erfc(z), torch.erfc(z)))
    return 0.5 * y


def log_ndtr(x, series_order=3, name="log_ndtr"):
    """Log Normal distribution function.
    For details of the Normal distribution function see `ndtr`.
    This function calculates `(log o ndtr)(x)` by either calling `log(ndtr(x))` or
    using an asymptotic series. Specifically:
    - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on
      `log(1-x) ~= -x, x << 1`.
    - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique
      and take a log.
    - For `x <= lower_segment`, we use the series approximation of erf to compute
      the log CDF directly.
    The `lower_segment` is set based on the precision of the input:
    ```
    lower_segment = { -20,  x.dtype=float64
                    { -10,  x.dtype=float32
    upper_segment = {   8,  x.dtype=float64
                    {   5,  x.dtype=float32
    ```
    When `x < lower_segment`, the `ndtr` asymptotic series approximation is:
    ```
       ndtr(x) = scale * (1 + sum) + R_N
       scale   = exp(-0.5 x**2) / (-x sqrt(2 pi))
       sum     = Sum{(-1)^n (2n-1)!! / (x**2)^n, n=1:N}
       R_N     = O(exp(-0.5 x**2) (2N+1)!! / |x|^{2N+3})
    ```
    where `(2n-1)!! = (2n-1) (2n-3) (2n-5) ...  (3) (1)` is a
    [double-factorial](https://en.wikipedia.org/wiki/Double_factorial).
    Args:
      x: `Tensor` of type `float32`, `float64`.
      series_order: Positive Python `integer`. Maximum depth to
        evaluate the asymptotic expansion. This is the `N` above.
      name: Python string. A name for the operation (default="log_ndtr").
    Returns:
      log_ndtr: `Tensor` with `dtype=x.dtype`.
    Raises:
      TypeError: if `x.dtype` is not handled.
      TypeError: if `series_order` is a not Python `integer.`
      ValueError:  if `series_order` is not in `[0, 30]`.
    """
    if not isinstance(series_order, int):
        raise TypeError("series_order must be a Python integer.")
    if series_order < 0:
        raise ValueError("series_order must be non-negative.")
    if series_order > 30:
        raise ValueError("series_order must be <= 30.")

    if x.dtype == torch.float64:
        lower_segment = torch.tensor(LOGNDTR_FLOAT64_LOWER, dtype=torch.float64, device=x.device)
        upper_segment = torch.tensor(LOGNDTR_FLOAT64_UPPER, dtype=torch.float64, device=x.device)
    elif x.dtype == torch.float32:
        lower_segment = torch.tensor(LOGNDTR_FLOAT32_LOWER, dtype=torch.float32, device=x.device)
        upper_segment = torch.tensor(LOGNDTR_FLOAT32_UPPER, dtype=torch.float32, device=x.device)
    else:
        raise TypeError("x.dtype=%s is not supported." % x.dtype)

    # The basic idea here was ported from:
    #   https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
    # We copy the main idea, with a few changes
    # * For x >> 1, and X ~ Normal(0, 1),
    #     Log[P[X < x]] = Log[1 - P[X < -x]] approx -P[X < -x],
    #     which extends the range of validity of this function.
    # * We use one fixed series_order for all of 'x', rather than adaptive.
    # * Our docstring properly reflects that this is an asymptotic series, not a
    #   Taylor series. We also provided a correct bound on the remainder.
    # * We need to use the max/min in the _log_ndtr_lower arg to avoid nan when
    #   x=0. This happens even though the branch is unchosen because when x=0
    #   the gradient of a select involves the calculation 1*dy+0*(-inf)=nan
    #   regardless of whether dy is finite. Note that the minimum is a NOP if
    #   the branch is chosen.
    return torch.where(
        x > upper_segment,
        -_ndtr(-x),  # log(1-x) ~= -x, x << 1
        torch.where(
            x > lower_segment,
            torch.log(_ndtr(torch.max(x, lower_segment))),
            _log_ndtr_lower(torch.min(x, lower_segment), series_order)))


def _log_ndtr_lower(x, series_order):
    """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
    x_2 = torch.square(x)
    # Log of the term multiplying (1 + sum)
    log_scale = (-0.5 * x_2 - torch.log(-x) - 0.5 * np.log(2. * np.pi))
    return log_scale + torch.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
    """Calculates the asymptotic series used in log_ndtr."""
    x_2 = torch.square(x)
    even_sum = torch.zeros_like(x)
    odd_sum = torch.zeros_like(x)
    x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
    for n in range(1, series_order + 1):
        y = _double_factorial(2 * n - 1) / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n = x_2n * x_2
    return 1. + even_sum - odd_sum


def _double_factorial(n):
    """The double factorial function for small Python integer `n`."""
    return np.prod(np.arange(n, 1, -2))

def ndtri(x):
    return torch.erfinv(2 * x - 1) * math.sqrt(2)

def log_sub_exp(x, y, return_sign=False):
    """Compute `log(exp(max(x, y)) - exp(min(x, y)))` in a numerically stable way.
    Use `return_sign=True` unless `x >= y`, since we can't represent a negative in
    log-space.
    Args:
      x: Float `Tensor` broadcastable with `y`.
      y: Float `Tensor` broadcastable with `x`.
      return_sign: Whether or not to return the second output value `sign`. If
        it is known that `x >= y`, this is unnecessary.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., `'log_sub_exp'`).
    Returns:
      logsubexp: Float `Tensor` of `log(exp(max(x, y)) - exp(min(x, y)))`.
      sign: Float `Tensor` +/-1 indicating the sign of `exp(x) - exp(y)`.
    """
    larger = torch.max(x, y)
    smaller = torch.min(x, y)
    result = larger + log1mexp(torch.max(larger - smaller, torch.tensor(0., dtype=larger.dtype, device=larger.device)))
    if return_sign:
        ones = torch.ones([], result.dtype)
        return result, torch.where(x < y, -ones, ones)
    return result


def log1mexp(x):
    """Compute `log(1 - exp(-|x|))` in a numerically stable way.
    Args:
      x: Float `Tensor`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., `'log1mexp'`).
    Returns:
      log1mexp: Float `Tensor` of `log1mexp(a)`.
    #### References
    [1]: Machler, Martin. Accurately computing log(1 - exp(-|a|))
         https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """

    x = torch.abs(x)
    return torch.where(
        # This switching point is recommended in [1].
        x < np.log(2), torch.log(-torch.expm1(-x)),
        torch.log1p(-torch.exp(-x)))


def _normal_log_pdf(value):
    return -(value ** 2) / 2 - math.log(math.sqrt(2 * math.pi))


def _normal_pdf(x, device):
    pi = torch.acos(torch.zeros(1))* 2 # which is 3.1415927410125732
    pi = pi.to(device)
    two_pi = 2 * pi #np.pi
    #print(two_pi, pi)
    return torch.rsqrt(two_pi) * torch.exp(-0.5 * torch.square(x))


def _normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def _standardized_low_and_high(loc=None,
                               scale=None,
                               low=None,
                               high=None):
    return (low - loc) / scale, (high - loc) / scale


def _normalizer(loc=None,
                scale=None,
                low=None,
                high=None,
                std_low=None,
                std_high=None):
    if std_low is None or std_high is None:
        std_low, std_high = _standardized_low_and_high(
            loc=loc, scale=scale, low=low, high=high)
    return ndtr(std_high) - ndtr(std_low)


def _log_normalizer(loc=None,
                    scale=None,
                    low=None,
                    high=None,
                    std_low=None,
                    std_high=None):
    if std_low is None or std_high is None:
        std_low, std_high = _standardized_low_and_high(
            loc=loc, scale=scale, low=low, high=high)
    #std_low, std_high = std_low.type(torch.float64), std_high.type(torch.float64)
    return log_sub_exp(log_ndtr(std_high), log_ndtr(std_low))#.type(torch.float32)

def _cdf(x, loc, scale, low, high):
    std_low, std_high = _standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    cdf_in_support = ((ndtr(
        (x - loc) / scale) - ndtr(std_low)) /
                      _normalizer(std_low=std_low, std_high=std_high))
    return torch.clamp(cdf_in_support, 0., 1.)

def _icdf(p, loc, scale, low, high):
    std_low, std_high = _standardized_low_and_high(
        low=low, high=high, loc=loc, scale=scale)
    std_low, std_high, p = std_low.type(torch.float64), std_high.type(torch.float64), p.type(torch.float64)
    low_cdf, high_cdf = ndtr(std_low), ndtr(std_high)
    std_icdf = ndtri(low_cdf + p * (high_cdf - low_cdf))
    return (std_icdf * scale + loc).type(torch.float32)


class TruncatedNormal(Distribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'low': constraints.real, 'high': constraints.real}
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, loc, scale, low=-1., high=1., validate_args=None, device=torch.device('cuda:1')):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)
        self.device = device
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(TruncatedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def _loc_scale_low_high(self):
        return self.loc, self.scale, self.low, self.high

    def log_prob(self, x):
        loc, scale, low, high = self._loc_scale_low_high()
        log_prob = -(0.5 * torch.square(
            (x - loc) / scale) + 0.5 * np.log(2. * np.pi) + torch.log(scale) + _log_normalizer(loc=loc, scale=scale, low=low, high=high))
        #xcpu = x.detach().cpu()
        #loccpu = loc.detach().cpu()
        #scalecpu = scale.detach().cpu()
        #lncpu = _log_normalizer(loc=loc, scale=scale, low=low, high=high).detach().cpu()
        #print(((torch.min(xcpu), torch.max(xcpu)), (torch.min(loccpu), torch.max(loccpu)), (torch.min(scalecpu), torch.max(scalecpu))))
        #print((torch.min(lncpu), torch.max(lncpu)))
        # p(x) is 0 outside the bounds.
        bounded_log_prob = torch.where((x > high) | (x < low),
                                    torch.ones_like(x) * (-np.inf),
                                    log_prob)
        #print((torch.min(log_prob.detach().cpu()), torch.max(log_prob.detach().cpu())))
        #print((torch.min(bounded_log_prob.detach().cpu()), torch.max(bounded_log_prob.detach().cpu())))
        return bounded_log_prob

    def prob(self, x):
        return torch.exp(self.log_prob(x))

    def cdf(self, x):
        loc, scale, low, high = self._loc_scale_low_high()
        return _cdf(x, loc, scale, low, high)

    def icdf(self, p):
        loc, scale, low, high = self._loc_scale_low_high()
        return _icdf(p, loc, scale, low, high)

    def log_cdf(self, x):
        loc, scale, low, high = self._loc_scale_low_high()
        std_low, std_high = _standardized_low_and_high(
            low=low, high=high, loc=loc, scale=scale)
        return (log_sub_exp(
            log_ndtr(
                (x - loc) / scale), log_ndtr(std_low)) -
                _log_normalizer(std_low=std_low, std_high=std_high))
    '''
    def entropy(self):
        loc, scale, low, high = self._loc_scale_low_high()
        std_low, std_high = _standardized_low_and_high(
            loc=loc, scale=scale, low=low, high=high)
        log_normalizer = _log_normalizer(std_low=std_low, std_high=std_high)
        return (
                0.5 * (1 + np.log(2.) + np.log(np.pi)) + torch.log(scale) +
                log_normalizer + 0.5 *
                (std_low * _normal_pdf(std_low) - std_high * _normal_pdf(std_high)) /
                torch.exp(log_normalizer))
    '''
    
    def entropy(self):
        loc, scale, low, high = self._loc_scale_low_high()
        std_low, std_high = _standardized_low_and_high(
            loc=loc, scale=scale, low=low, high=high)
        log_normalizer = _log_normalizer(std_low=std_low, std_high=std_high)
        pi = torch.acos(torch.zeros(1))* 2
        pi = pi.to(self.device)
        return (
                0.5 * (1 + torch.log(torch.tensor(2.).to(self.device)) + torch.log(pi)) + torch.log(scale) +
                log_normalizer + 0.5 *
                (std_low * _normal_pdf(std_low, self.device) - std_high * _normal_pdf(std_high, self.device)) /
                torch.exp(log_normalizer))
    @property
    def mean(self):
        loc, scale, low, high = self._loc_scale_low_high()
        std_low, std_high = _standardized_low_and_high(
            loc=loc, scale=scale, low=low, high=high)
        lse, sign = log_sub_exp(_normal_log_pdf(std_low),
                                 _normal_log_pdf(std_high),
                                 return_sign=True)
        return loc + scale * sign * torch.exp(
            lse - _log_normalizer(std_low=std_low, std_high=std_high))


    @property
    def variance(self):
        loc, scale, low, high = self._loc_scale_low_high()
        std_low, std_high = _standardized_low_and_high(
            loc=loc, scale=scale, low=low, high=high)
        log_normalizer = _log_normalizer(std_low=std_low, std_high=std_high)
        var = (
                torch.square(scale) *
                (1. +
                 (std_low * _normal_pdf(std_low,self.device) - std_high * _normal_pdf(std_high,self.device)) /
                 torch.exp(log_normalizer) -
                 torch.exp(2. * (
                         log_sub_exp(  # ignore sign because result gets squared
                             _normal_log_pdf(std_low), _normal_log_pdf(std_high))
                         - log_normalizer))))
        return var

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(TruncatedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            sample = torch.rand(*shape, dtype=self.loc.dtype, device=self.loc.device)
            sample = torch.clamp(sample, 0.001, 0.999)
            output = self.icdf(sample)
            #print(max(output), min(output))

            output = torch.clamp(output, -0.999, 0.999)
            return output

    def rsample(self, sample_shape=torch.Size()):
        loc, scale, low, high = self._loc_scale_low_high()
        shape = self._extended_shape(sample_shape)
        sample = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return _icdf(sample, loc, scale, low, high)

class DiscNormal(TruncatedNormal):
    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= -1.0

        means, log_scales = self.loc, torch.log(self.scale)
        centered = samples - means                                         # B, 3, H, W
        inv_stdv = torch.exp(- log_scales)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = probit(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = probit(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(127.5))
        # woow the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 3, H, W

        return log_probs

