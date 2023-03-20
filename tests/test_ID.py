#!/usr/bin/env python
import unittest
import argparse
import numpy as np

from libraries import identify_diffusion as ID_library

class TestIdentifyDiffusions(unittest.TestCase):
    """Class for testing the identification of hoppings.
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
    
    def test_POTIM_missing(self):
        """Checks that exits if POTIM is missing.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/INCAR_POTIM_missing'])
        
        with self.assertRaises(SystemExit):
            inp = ID_library.xdatcar(args)
    
    def test_NBLOCK_missing(self):
        """Checks that exits if NBLOCK is missing.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/INCAR_NBLOCK_missing'])
        
        with self.assertRaises(SystemExit):
            inp = ID_library.xdatcar(args)
    
    # XDATCAR file reading
    
    def test_parameters_reading(self):
        """Checks that defining parameters are read correctly.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/XDATCAR_ok'])
        inp = ID_library.xdatcar(args)
        
        # Simulation box
        expected_cell = np.array([[13.2, 0,    0],
                                  [0,    13.2, 0],
                                  [0,    0,    25.3]])
        
        self.assertEqual(inp.cell.tolist(), expected_cell.tolist())
        self.assertEqual(inp.TypeName,      ['Li', 'La', 'Zr', 'O'])
        self.assertEqual(inp.Nelem.tolist(), [110,  48,   32,   192])
        self.assertEqual(inp.position[0, 0, 0], 0.5)
    
    def test_scale_wrong(self):
        """Checks that exits if the scale is wrongly defined.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/XDATCAR_scale_wrong'])
        with self.assertRaises(SystemExit):
            inp = ID_library.xdatcar(args)
    
    def test_cell_wrong(self):
        """Checks that exits if the cell is wrongly defined.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/XDATCAR_cell_wrong'])
        with self.assertRaises(SystemExit):
            inp = ID_library.xdatcar(args)
    
    def test_composition_wrong(self):
        """Checks that exits if the composition is wrongly defined.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/XDATCAR_composition_wrong'])
        with self.assertRaises(SystemExit):
            inp = ID_library.xdatcar(args)
    
    def test_pos_wrong(self):
        """Checks that exits if the configurations is wrongly defined.
        """
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--MD_path')
        args = parser.parse_args(['--MD_path', 'tests/data/XDATCAR_pos_wrong'])
        with self.assertRaises(SystemExit):
            inp = ID_library.xdatcar(args)
