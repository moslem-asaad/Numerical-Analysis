"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
import assignment2

class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.f = None
        self.f1= None
        self.f2 = None
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        self.f = f
        if n%2 != 0:
            return self.simpson_rule(f,a,b,n-1)
        else:
            return self.trapezoidal(f,a,b,n-1)


    def trapezoidal(self,f, a, b, n):
        dx = (b - a) / n
        s = 0.0
        s += f(a) / 2.0
        for i in range(1, n):
            s += f(a + i * dx)
        s += f(b) / 2.0
        return np.float32(s * dx)

    def simpson_rule(self,f,a,b,n):
        dx = (b - a) / n
        val = f(a) + f(b)
        for i in range(1, n):
            if i % 2 != 0:
                val += 4 * f(a + i * dx)
            else:
                val += 2 * f(a + i * dx)
        return np.float32(dx / 3 * val)


    # replace this line with your solution

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        self.f1 = f1
        self.f2 = f2
        intersection_points = assignment2.Assignment2().intersections(f1,f2,1,100)
        num_of_intersections = len(intersection_points)
        if num_of_intersections <=1 :
            return np.nan
        else:
            area = 0
            for i in range(0,num_of_intersections-1):
                curr_point = intersection_points[i]
                next_point = intersection_points[i+1]
                some_point = (next_point - curr_point) / 2
                area+= self.g(curr_point,next_point,some_point)

        return np.float32(area)

    def g(self,curr_point, next_point, point):
        #x = self.f2(point) - self.f1(point)
        if self.f1(point) >= self.f2(point):
            f = lambda x : self.f1(x) - self.f2(x)
        else:
            f = lambda x : self.f2(x) - self.f1(x)
        return abs(self.integrate(f, curr_point, next_point, 5000))



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        #f1 = strong_oscilations()
        f1 = np.poly1d([-371,123,-15,3,-5,8])
        #print(f1)
        #r = ass3.integrate(f1, 0.09, 10, 20)
        #r = ass3.integrate(f1, 0.0, 1.0, 11)
        for i in range(100,500):
            r = ass3.integrate(f1, -1.0, 1.0, i)
            true_result = 67.2
            #print(r)
            #print(i)
            self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

        #true_result = 0.5
        #true_result = -7.78662 * 10 ** 33
        #self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_between(self):
        ass3 = Assignment3()
        #ff1 = np.poly1d([1,-2,0,1])
        #ff2 = np.poly1d([1,0])
        def f10(x):
            return np.sin(log(x))

        def f6(x):
            return np.sin(x) / x - 0.1
        true_result = 2.76555
        area = ass3.areabetween(f10,f6)
        print(area)
        self.assertGreaterEqual(0.001, abs((area - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
