"""
Basic tests for the FastMatchedFilter package.
"""

import unittest
import os
import numpy as np

from fast_matched_filter import matched_filter


class TestFastMatchedFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Read in the data - do this only once to save time."""
        cls.test_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'data')
        cls.kaik = {
            'data': np.load(
                os.path.join(cls.test_path, 'kaik_data.npy')),
            'templates':  np.load(os.path.join(
                cls.test_path, 'kaik_template.npy')),
            'pads': np.load(os.path.join(
                cls.test_path, 'kaik_pads.npy'))}
        cls.noisy = {
            'data': np.load(os.path.join(
                cls.test_path, 'noisy_data.npy')),
            'templates': np.load(os.path.join(
                cls.test_path, 'noisy_template.npy')),
            'pads': np.load(os.path.join(
                cls.test_path, 'noisy_pads.npy'))}

        cls.cccsums = {}
        for dataset, name in zip([cls.kaik, cls.noisy], ['kaik', 'noisy']):
            print("\nComputing correlations for dataset %s" % name)
            weights = np.ones(
                (len(dataset['templates']), len(dataset['data'])))
            for arch in ['cpu', 'gpu']:
                print("Trying to compute using the %s" % arch)
                cccsums = matched_filter(
                    templates=dataset['templates'], weights=weights,
                    moveouts=dataset['pads'], data=dataset['data'],
                    step=1, arch=arch)
                if cccsums is None:
                    # This should only happen if something isn't compiled
                    print("Cannot test for architecture %s" % arch)
                    continue
                else:
                    cls.cccsums.update({name + '_' + arch: cccsums})

    def test_no_nans(self):
        """Check that no strange values exist cross-correlations"""
        nan_status = {}
        for key in self.cccsums.keys():
            print('NaN checking for: %s' % key)
            status = np.any(np.isnan(self.cccsums[key]))
            if status:
                print("Found NaN values for %s" % key)
            nan_status.update({key: status})
        for key in nan_status.keys():
            self.assertFalse(nan_status.key)

    def test_within_bounds(self):
        """Check that correlations are between +/- 1"""
        maxima = []
        minima = []
        for key in self.cccsums.keys():
            print("Checking reasonable boundedness for correlations for"
                  " %s" % key)
            print("Max of data is: %f" % np.max(self.cccsums[key]))
            print("Min of data is: %f" % np.min(self.cccsums[key]))
            maxima.append(np.max(self.cccsums[key]))
            minima.append(np.min(self.cccsums[key]))
        self.assertTrue(np.all(maxima <= 1.0))
        self.assertTrue(np.all(minima >= -1.0))


if __name__ == '__main__':
    unittest.main()
