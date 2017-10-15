"""
Minimal setup file for the fast_matched_filter library for Python packaging.

:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import os
from setuptools import setup
from setuptools.command.install import install
from distutils.command.build import build
from subprocess import call


# Define a new build command
class FastMatchedFilterBuild(build):
    def run(self):
        # Run standard python build things
        build.run(self)
        # Build the Python libraries via Makefile
        cpu_make = ['make', 'python_cpu']
        gpu_make = ['make', 'python_gpu']

        gpu_built = False
        cpu_built = False

        ret = call(cpu_make)
        if ret == 0:
            cpu_built = True
        ret = call(gpu_make)
        if ret == 0:
            gpu_built = True
        if gpu_built is False:
            print("Could not build GPU code")
        if cpu_built is False:
            raise OSError("Could not build cpu code")


# Define a new install command
class FastMatchedFilterInstall(install):
    def run(self):
        install.run(self)


# Get the long description - it won't have md formatting properly without
# using pandoc though, but that adds another dependancy.
with open('README.md') as f:
    long_description = f.read()

setup(name='FastMatchedFilter',
      version='0.0.1',
      description='Fast time-domain normalised cross-correlation for '
                  'CPU & GPU',
      long_description=long_description,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: GPL License',
        'Programming Language :: Python :: 2.7, 3.5, 3.6',
        'Topic :: Seismology :: Matched Filter',
      ],
      url='https://github.com/beridel/fast_matched_filter',
      author='William Frank',
      author_email='',
      license='GPL',
      packages=['fast_matched_filter'],
      install_requires=['numpy'],
      tests_require=['pytest>=2.0.0'],
      include_package_data=True,
      zip_safe=False,
      cmdclass={
          'build': FastMatchedFilterBuild,
          'install': FastMatchedFilterInstall})
