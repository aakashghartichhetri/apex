import unittest
from apex.phase02 import *

class Phase02Tests(unittest.TestCase):
    def test_memory(self):
        self.assertEqual(candidates((Device("tiny",1,1),)),[])
    def test_reproducible(self):
        self.assertEqual(run("apex",7,100),run("apex",7,100))
    def test_bounds(self):
        for p in POLICIES:
            r=run(p,1,100)
            for k in ("accuracy","miss_rate","cpu_util","accelerator_util"):
                self.assertTrue(0<=r[k]<=1)
    def test_queue(self):
        d,m=choose("queue",candidates(),{"cpu":0,"accelerator":100},0,.5,1)
        self.assertEqual(d.name,"cpu")
    def test_bad_n(self):
        with self.assertRaises(ValueError): run("apex",n=0)
