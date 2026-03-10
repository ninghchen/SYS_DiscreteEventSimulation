import math
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import skew


def ConstructCI(dataArray, alpha=0.05, ci_type='z'):
    """Synopsis: Constructs a (symmetric) confidence interval of the
    specified type for the dataArray using significance level alpha.

    Arguments:

    - dataArray: a list of floats

      Will throw an exception if len(dataArray) <= 1.

    - alpha: significance level (alpha) in [0,1] (default 0.05)

    - ci_type: the type of confidence interval. The only permissible
      values are:

      'z': Use the standard normal distribution (default)
      't': Use the Student-t distribution
      'w': Use the Willink interval construction

      Will throw an exception if type is invalid.

    Returns:
    - A four-tuple CI-min, CI-max, x-bar, and half-length

      Note: For the Willink interval, the returned half length is
      1/2 of the width of the interval (i.e., 0.5*(CI-max - CI-min)).
    """

    n = len(dataArray)
    if n <= 1:
        raise ValueError(f'ConstructCI: invalid n {n}')

    percentile = 0.0

    if ci_type == 'z':
        percentile = norm.ppf(1.0 - 0.5 * alpha, loc=0.0, scale=1.0)
    elif ci_type == 't':
        percentile = t.ppf(1.0 - 0.5 * alpha, n - 1, loc=0.0, scale=1.0)
    elif ci_type == 'w':
        percentile = t.ppf(1.0 - 0.5 * alpha, n - 1, loc=0.0, scale=1.0)
    else:
        raise ValueError(f'ConstructCI: invalid ci_type {ci_type}')

    xBar_n = np.average(dataArray)
    S2_n = np.var(dataArray, ddof=1)

    if ci_type == 'w':

        # If you work out the details, you will find that the
        # quantity g-1 in scipy.skew documentation has the
        # following relation with the quantity 'a' in Law's
        # formulation of the Willink interval:
        #
        # a = math.sqrt(n-1)/(6.0*(N-2.0)) * g-1

        def G(a, r):
            # Note: caution is needed for cube root of negatives.

            rad = 1.0 + 6.0 * a * (r - a)
            if rad >= 0:
                return (0.5 / a) * (math.pow(rad, 1.0 / 3.0) - 1.0)
            else:
                return (0.5 / a) * (-math.pow(-rad, 1.0 / 3.0) - 1.0)

        a = math.sqrt(n - 1) / (6.0 * (n - 2.0)) * skew(dataArray)
        g_upper = G(a, -percentile)
        g_lower = G(a, percentile)
        CI_min = xBar_n - g_lower * math.sqrt(S2_n / float(n))
        CI_max = xBar_n - g_upper * math.sqrt(S2_n / float(n))
        halfLength = 0.5 * (CI_max - CI_min)

        return CI_min, CI_max, xBar_n, halfLength

    else:  # 'z' and 't'
        halfLength = percentile * math.sqrt(S2_n / float(n))
        CI_min = xBar_n - halfLength
        CI_max = xBar_n + halfLength

        return CI_min, CI_max, xBar_n, halfLength



def Law_2013_Example_4_27(alpha=0.10):
    """Example 4.27 from Law (2013)."""

    Z = [1.20, 1.50, 1.68, 1.89, 0.95, 1.49, 1.58, 1.55, 0.50, 1.09]

    for t in ['z', 't', 'w']:
        minV, maxV, xBar, hLength = ConstructCI(Z, alpha=0.10, ci_type=t)
        print(f'CI type {t}  CI: [{minV:.3f}, {maxV:.3f}]' +
              f' xBar={xBar:.3f}  hLength={hLength:.3f}')


if __name__ == "__main__":
    # Note that Z in Law's example is the same as Z in HW2 Problem 1
    Law_2013_Example_4_27(alpha=0.05)


    