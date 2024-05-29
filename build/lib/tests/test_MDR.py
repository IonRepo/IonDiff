#!/usr/bin/env python
import unittest
import argparse
import numpy as np

from libraries import common_library as CL_library

class TestIdentifyDiffusions(unittest.TestCase):
    """Class for testing the reading of the molecular dynamics file.
    """
    
    # XDATCAR file reading
    
    def test_parameters_reading(self):
        """Checks that defining parameters are read correctly.
        """
        
        cell, n_ions, compounds, concentration, coordinates = CL_library.read_XDATCAR('tests/data/XDATCAR_ok')
        
        # Simulation box
        expected_cell = np.array([[13.2, 0,    0],
                                  [0,    13.2, 0],
                                  [0,    0,    25.3]])
        
        self.assertEqual(cell.tolist(), expected_cell.tolist())
        self.assertEqual(compounds,              ['Li', 'La', 'Zr', 'O'])
        self.assertEqual(concentration.tolist(), [110,  48,   32,   192])
        self.assertEqual(coordinates[0, 0, 0], 0.5)
    
    def test_scale_wrong(self):
        """Checks that exits if the scale is wrongly defined.
        """
        
        with self.assertRaises(SystemExit):
            CL_library.read_XDATCAR('tests/data/XDATCAR_scale_wrong')
    
    def test_cell_wrong(self):
        """Checks that exits if the cell is wrongly defined.
        """
        
        with self.assertRaises(SystemExit):
            CL_library.read_XDATCAR('tests/data/XDATCAR_cell_wrong')
    
    def test_composition_wrong(self):
        """Checks that exits if the composition is wrongly defined.
        """
        
        with self.assertRaises(SystemExit):
            CL_library.read_XDATCAR('tests/data/XDATCAR_composition_wrong')
    
    def test_pos_wrong(self):
        """Checks that exits if the positions are wrongly defined.
        """
        
        with self.assertRaises(SystemExit):
            CL_library.read_XDATCAR('tests/data/XDATCAR_pos_wrong')

