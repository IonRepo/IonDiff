#!/usr/bin/env python
import unittest
import argparse
import numpy as np

from libraries import common_library as CL_library

class TestIdentifyDiffusions(unittest.TestCase):
    """Class for testing the reading of simulation parameters.
    """
    
    # INCAR file reading
    
    def test_both_ok(self):
        """Checks that both flags are read correctly in a prototypical file.
        """
        
        delta_t, n_steps = CL_library.read_INCAR('tests/data/INCAR_both_ok')
        self.assertEqual(delta_t, 1.5)
        self.assertEqual(n_steps, 10)
    
    def test_POTIM_missing(self):
        """Checks that exits if POTIM is missing.
        """
        
        with self.assertRaises(SystemExit):
            inp = CL_library.read_INCAR('tests/data/INCAR_POTIM_missing')
    
    def test_NBLOCK_missing(self):
        """Checks that exits if NBLOCK is missing.
        """
        
        with self.assertRaises(SystemExit):
            inp = CL_library.read_INCAR('tests/data/INCAR_NBLOCK_missing')
