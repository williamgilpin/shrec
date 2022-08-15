"""
To run all tests from the top-level directory
> python -m unittest
"""
#!/usr/bin/env python
import os
import numpy as np
import unittest

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKING_DIR)

import sys

sys.path.insert(1, os.path.join(WORKING_DIR, "shrec"))
from shrec.models import RecurrenceClustering


class TestDiscrete(unittest.TestCase):
    """
    Tests discrete model
    """
    def test_discrete_model(self):
        """
        Test loading the model
        """

        model = RecurrenceClustering(resolution=1.0, tolerance=0.01, random_state=1)
        np.random.seed(0)
        data = np.tile([np.sin(np.linspace(0, 2, 300))], (3, 1)).T

        label_vals = model.fit_predict(data)
        self.assertEqual(
            len(label_vals),  
            data.shape[0], 
            "Model did not predict all data points"
        )

        
        
if __name__ == "__main__":
    unittest.main()