#!/usr/bin/env python
import unittest
import argparse
import numpy as np

from libraries import identify_diffusion as ID_library

class TestIdentifyDiffusions(unittest.TestCase):
    """Class for testing the analysis of correlations.
    """
    
    # INCAR file reading
    
    def test_both_ok(self):
        """Checks that both flags are read correctly in a prototypical file.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/INCAR_both_ok'])
        inp = ID_library.xdatcar(args)
        
        delta_t, n_steps = inp.read_INCAR(args)
        self.assertEqual(delta_t, 1.5)
        self.assertEqual(n_steps, 10)
