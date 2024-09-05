"""
In this assignment you should interpolate the given function.
"""
import bisect
import math
import operator
from functools import reduce

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """
        self.xpoints = []
        self.ypoints = []
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.c_tag = []
        self.d_tag = []
        self.coeffeicents = None
        self.dx_arr = []
    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        self.xpoints = []
        self.ypoints = []
        self.make_points(f, a, b, n)
        if n < 3 :
            return lambda x : self.lagrange_method(x)
        else:
            self.find_coffs(len(self.xpoints))
            return lambda x : self.spline_eval(x)
        # replace this line with your solution to pass the second test

    def make_points(self, f, a, b, n):
        interval = (b - a) / n
        for i in range(0, n):
            try:
                x = a + i * interval
                y = f(x)
                self.xpoints.append(x)
                self.ypoints.append(y)
            except: print()
            finally:
                continue

    def lagrange_method(self, x):
        def mul_iter(j):
            mul = [(x - self.xpoints[j]) / (self.xpoints[j] - self.xpoints[i]) for i in range(0, n) if j != i]
            return reduce(operator.mul, mul)

        assert (len(self.xpoints) != 0 and len(self.xpoints) == len(self.ypoints))
        n = (len(self.xpoints))
        result = sum(mul_iter(j) * self.ypoints[j] for j in range(0, n))
        return result

    def compute_A(self, n, dx):
        a = np.zeros(n - 1)
        for i in range(0, n - 2):
            dx_i = dx[i]
            a[i] = dx_i / (dx_i + dx[i + 1])
        self.A = a

    def compute_B(self, n):
        b = [2] * n
        self.B = b

    def compute_C(self, n, dx):
        c = np.zeros(n - 1)
        for i in range(0, n - 2):
            dx_i_plus = dx[i + 1]
            c[i+1] = dx_i_plus / (dx_i_plus + dx[i])
        self.C = c

    def compute_right_side(self, n, dx):
        d = np.zeros(n)
        for i in range(1, n - 1):
            dy1dxi = (self.ypoints[i + 1] - self.ypoints[i]) / dx[i]
            dy2dxim = (self.ypoints[i] - self.ypoints[i - 1]) / dx[i - 1]
            d[i] = 6 * (dy1dxi - dy2dxim) / (dx[i] + dx[i - 1])
        self.D = d

    def compute_c_tag(self, n):
        self.c_tag = np.zeros(n)
        self.c_tag[0] = self.C[0] / self.B[0]
        for i in range(1, n-1):
            numerator = self.C[i]
            denumerator = self.B[i] - self.A[i - 1] * self.c_tag[i - 1]
            self.c_tag[i] = numerator/denumerator

    def compute_d_tag(self, n):
        self.d_tag = np.zeros(n)
        self.d_tag[0] = self.D[0] / self.B[0]
        for i in range(1, n):
            numerator = self.D[i] - self.d_tag[i - 1] * self.A[i - 1]
            denumenator = self.B[i] - self.c_tag[i - 1] * self.A[i - 1]
            self.d_tag[i] = numerator / denumenator

    def ThomasAlgorithm(self, n):
        result = np.zeros(n)
        result[n-1] = self.d_tag[n-1]
        for i in range(n - 2, -1, -1):
            result[i] = result[i + 1] * self.c_tag[i] + self.d_tag[i]
        return result

    def find_coffs(self, n):
        dx = np.zeros(n - 1)
        for i in range(0, n - 1):
            dx[i] = self.xpoints[i + 1] - self.xpoints[i]
        self.compute_A(n, dx)
        self.compute_B(n)
        self.compute_C(n, dx)
        self.compute_right_side(n, dx)
        self.compute_c_tag(n)
        self.compute_d_tag(n)
        thomas_res = self.ThomasAlgorithm(n)
        coeffs = [[], [], [], []]
        coeffs[0] = np.zeros(n - 1)
        coeffs[1] = np.zeros(n - 1)
        coeffs[2] = np.zeros(n - 1)
        coeffs[3] = np.zeros(n - 1)
        for i in range(0, n - 1):
            ho = (thomas_res[i + 1] - thomas_res[i]) * dx[i] / 6
            coeffs[0][i] = ho
            sec_order = (thomas_res[i] * dx[i] ** 2) / 2
            coeffs[1][i] = sec_order
            one_order = (self.ypoints[i + 1] - self.ypoints[i] - (thomas_res[i + 1] + 2 * thomas_res[i]) * dx[i] ** 2) / 6
            coeffs[2][i] = one_order
            coeffs[3][i] = self.ypoints[i]
        self.coeffeicents = coeffs
        self.dx_arr = dx

    def binary_search(self,arr,low,x,high):
        while low<high:
            mid = (low + high)//2
            if arr[mid] < x:
                low = mid  + 1
            else :
                high = mid
        if low < len(arr):
            return low
        else:
            return len(arr) - 1

    def spline_eval(self, x):
        j = self.binary_search(self.xpoints,0,x,len(self.xpoints)) - 1
        theta = (x - self.xpoints[j]) / self.dx_arr[j]
        result = self.coeffeicents[3][j] + theta * \
                 (self.coeffeicents[2][j] + theta * (self.coeffeicents[1][j] + theta * self.coeffeicents[0][j]))
        return result
##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)
            f = np.poly1d([5])
            #def f(q):
            #    return np.arctan(q)
            #def f(x):
            #    return pow(x, 2) - 3 * x + 2
            #R = RESTRICT_INVOCATIONS
            #def f(q):
            #    return R(10)(f2(q))
            ff = ass1.interpolate(f, -10, 10, 5)

            xs = np.random.random(10)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                # print("ff",yy)
                # print("f",y)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
