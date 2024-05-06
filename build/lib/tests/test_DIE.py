#!/usr/bin/env python
import unittest
import argparse
import numpy as np

from libraries import common_library as CL_library

class TestIdentifyDiffusions(unittest.TestCase):
    """Class for testing the extraction of diffusive information from chemical formula.
    """
    
    # Diffusive information extraction
    
    def test_obtain_diffusive_information(self):
        """Checks that diffusive the information extracted from the name is correct.
        """
        
        DiffTypeName, NonDiffTypeName, _ = CL_library.obtain_diffusive_information(['C', 'H', 'N', 'Pb', 'Br'], [1, 6, 1, 1, 3])
        
        self.assertEqual(DiffTypeName, ['Br'])  # Diffusive species
        self.assertEqual(NonDiffTypeName, ['C', 'H', 'N', 'Pb'])  # Non-diffusive species
    
    def test_obtain_diffusive_family(self):
        """Checks that diffusive families are recognized correctly.
        """
        
        self.assertEqual(CL_library.obtain_diffusive_family('Cu2Se'),        'Cu-based')
        self.assertEqual(CL_library.obtain_diffusive_family('Li10GeS2P12'),  'Li-based')
        self.assertEqual(CL_library.obtain_diffusive_family('CH3NH3-PbBr3'), 'Halide-based')
        self.assertEqual(CL_library.obtain_diffusive_family('SrCoO3'),       'O-based')
        self.assertEqual(CL_library.obtain_diffusive_family('AgCrSe2'),      'Ag-based')
        self.assertEqual(CL_library.obtain_diffusive_family('NaBO2'),        'Na-based')
