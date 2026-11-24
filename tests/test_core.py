import unittest,random
from apex.core import *
from apex.live import answer
class TestCore(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(workload(100,1),workload(100,1))
    def test_calibration(self):
        c=Calibration().fit(workload(500,1));self.assertTrue(all(0<x<1 for x in c.predict(.5)))
    def test_fixed(self):
        r=workload(100,1);c=Calibration().fit(workload(100,2))
        self.assertEqual(evaluate(r,c,'small')['large_fraction'],0)
        self.assertEqual(evaluate(r,c,'large')['large_fraction'],1)
    def test_congestion(self):
        self.assertEqual(choose('apex',(.3,.99),(0,100),(.1,.4),random.Random(1)),0)
    def test_metrics(self):
        r=workload(100,1);c=Calibration().fit(workload(100,2))
        for p in POLICIES:
            m=evaluate(r,c,p);self.assertTrue(0<=m['accuracy']<=1);self.assertTrue(0<=m['deadline_miss']<=1)
    def test_parser(self):
        self.assertEqual(answer('steps #### 1,234'),'1234');self.assertIsNone(answer('no answer'))
if __name__=='__main__':unittest.main()
