"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution

        roots = []
        #base_func = f2 - f1
        while 1:
            low , high = self.find_range(a,b,f1,f2)
            if low is None:
                break
            a = high
            root = self.bisection_method(low,high,f1,f2)
            i=0
            if root != None :
                if i == 0:
                    roots.append(root)
                    i+=1
                else:
                    if abs(roots[i] - root) > maxerr:
                        roots.append(root)
                        i += 1
        return roots


    def find_range(self ,a ,b,f1,f2):
        tol = 0.0001
        low = a
        high = a + tol
        fa = f2(a) - f1(a)
        fb = f2(high) - f1(high)
        #fa = base_func(a)
        #fb = base_func(high)
        while fa*fb > 0 :
            if low < b:
                low = high
                high = high + tol
                fa = fb
                fb = f2(high) - f1(high)
                #fb = base_func(high)
            else: return None,None
        return low,high

    def bisection_method(self,a, b,f1,f2):
        fa = f2(a) - f1(a)
        fb = f2(b) - f1(b)
        if fa * fb > 0:
            return None
        z = (a + b) / 2
        while abs(a - b) >= 0.0002:
            z = (a + b) / 2
            fz = f2(z) - f1(z)
            if fz == 0.0:
                break
            fa = f2(a) - f1(a)
            fb = f2(b) - f1(b)
            if fa * fb < 0:
                b = z
            else:
                a = z
        return z




##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        #f1 = np.poly1d([-1, 0, 1])
        #f2 = np.poly1d([1, 0, -1])
        def f1(x):
            return np.sin(log(x))
        def f2(x):
            return pow(x, 2) - 3 * x + 2
        X = ass2.intersections(f1, f2, 1, 100, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
