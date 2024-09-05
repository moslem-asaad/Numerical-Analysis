"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""
import operator
from functools import reduce

import numpy as np
import time
import random


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.pol = None
        self.xpoints = []
        self.ypoints = []

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        self.make_points(a,b,maxtime,f)
        self.poly_fit(np.array(self.xpoints), self.ypoints, d)
        return lambda x : self.eval_poly(x,self.pol)

        # replace these lines with your solution

    def make_points(self,low,high,maxtime,f):
        in_time = time.time()
        process_time = maxtime * 0.15
        making_points_time = maxtime - process_time
        while making_points_time >= time.time() - in_time:
            x = random.uniform(low,high)
            y = f(x)
            self.xpoints.append(x)
            self.ypoints.append(y)

    def build_Coff_matrix(self, deg, x):
        # h = theta + theta * x + theta * x^2 + ..
        # matrix = 0 , x , x^2 , ...
        matrix = np.zeros(shape=(x.shape[0], deg + 1))
        ones = np.ones_like(x)
        for i in range(0, deg + 1):
            if i == 0:
                matrix[:, 0] = ones
            if i == 1:
                matrix[:, 1] = x
            else:
                matrix[:, i] = pow(x, i)
        return matrix

    def poly_fit(self, xpoints, ypoints, deg):
        coff_matrix = self.build_Coff_matrix(deg, xpoints)
        pol = self.fit_x(coff_matrix, ypoints)
        self.pol = pol[::-1]

    def fit_x(self, mat, y):
        solv = np.linalg.lstsq(mat, y,rcond=None)
        return solv[0]

    def eval_poly(self, x, pol):
        y = 0
        for p in pol:
            y *= x
            y += p
        return y


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            #print("fx ", f(x))
            #print("ffx ", ff(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
