"""
Basic tests for the FastMatchedFilter package.
"""

import unittest
import os
import pytest
import numpy as np

from fast_matched_filter import (matched_filter, CPU_LOADED, GPU_LOADED)


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
        cls.datasets = ['noisy', 'kaik']

        cls.cccsums = {}
        for dataset, name in zip([cls.kaik, cls.noisy], cls.datasets):
            print("\nComputing correlations for dataset %s" % name)
            weights = np.ones(
                (len(dataset['templates']), len(dataset['data'])))
            weights /= len(dataset['data'])
            for arch in ['cpu', 'gpu']:
                print("Trying to compute using the %s" % arch)
                cccsum = matched_filter(
                    templates=dataset['templates'], weights=weights,
                    moveouts=dataset['pads'], data=dataset['data'],
                    step=1, arch=arch)
                if cccsum is None:
                    # This should only happen if something isn't compiled
                    print("Cannot test for architecture %s" % arch)
                    continue
                else:
                    cls.cccsums.update({name + '_' + arch: cccsum})

                print("Trying to compute with single template format using the %s" % arch)
                cccsum = matched_filter(
                    templates=dataset['templates'][0, :, :, :], weights=weights[0, :],
                    moveouts=dataset['pads'][0, :], data=dataset['data'],
                    step=1, arch=arch)
                if cccsum is None:
                    # This should only happen if something isn't compiled
                    print("Cannot test for architecture %s" % arch)
                    continue
                else:
                    cls.cccsums.update({'single_' + name + '_' + arch: cccsum})
                
                print("Trying to compute with traces format using the %s" % arch)
                n_templates = dataset['templates'].shape[0]
                n_stations = dataset['templates'].shape[1]
                n_components = dataset['templates'].shape[2]
                n_samples_template = dataset['templates'].shape[3]
                n_samples_data = dataset['data'].shape[-1]

                templates = dataset['templates'].reshape(
                    n_templates, n_stations * n_components, n_samples_template)
                weights_alt = weights.reshape(
                    n_templates, n_stations * n_components)
                moveouts = dataset['pads'].reshape(
                    n_templates, n_stations * n_components)
                data = dataset['data'].reshape(
                    n_stations * n_components, n_samples_data)
                cccsum = matched_filter(
                    templates=templates, weights=weights_alt,
                    moveouts=moveouts, data=data,
                    step=1, arch=arch)
                if cccsum is None:
                    # This should only happen if something isn't compiled
                    print("Cannot test for architecture %s" % arch)
                    continue
                else:
                    cls.cccsums.update({'traces_' + name + '_' + arch: cccsum})

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
            self.assertFalse(nan_status[key])

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
        self.assertTrue(np.all(np.array(maxima) <= 1.0))
        self.assertTrue(np.all(np.array(minima) >= -1.0))

    @pytest.mark.skipif(CPU_LOADED is False or GPU_LOADED is False,
                        reason="Either CPU or GPU have not run")
    def test_compare_gpu_cpu(self):
        tolerance = 0.001
        for dataset in self.datasets:
            print("Comparing for {dataset}".format(dataset=dataset))
            if not np.allclose(self.cccsums[dataset + '_gpu'],
                               self.cccsums[dataset + '_cpu'], atol=tolerance):
                print("GPU and CPU are not similar at {tolerance}. "
                      "Maximum difference is {diff}".format(
                            diff=np.abs(self.cccsums[dataset + '_cpu'] -
                                        self.cccsums[dataset + '_gpu']).max(),
                            tolerance=tolerance))
            self.assertTrue(np.allclose(
                self.cccsums[dataset + '_gpu'], self.cccsums[dataset + '_cpu'],
                atol=tolerance))

    def test_single(self):
        print("Checking that single template input is reformatted correctly")
        for dataset in self.datasets:
            for arch in ['cpu', 'gpu']:
                self.assertTrue(np.allclose(
                    self.cccsums['single_' + dataset + '_' + arch], 
                    self.cccsums[dataset + '_' + arch],
                    atol=0.0001))
                    
    def test_format(self):
        print("Checking that [traces] and [stations x components] formats are consistent")
        for dataset in self.datasets:
            for arch in ['cpu', 'gpu']:
                self.assertTrue(np.allclose(
                    self.cccsums['traces_' + dataset + '_' + arch],
                    self.cccsums[dataset + '_' + arch],
                    atol=0.0001))


if __name__ == '__main__':
    unittest.main()
