"""
Minimal setup file for the fast_matched_filter library for Python packaging.

:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call


class FMFExtension(Extension):
    def __init__(self, name):
        # Don't run the default setup-tools build commands, use the custom one
        Extension.__init__(self, name=name, sources=[])


# Define a new build command
class FastMatchedFilterBuild(build_ext_original):
    def run(self):
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


# Get the long description - it won't have md formatting properly without
# using pandoc though, but that adds another dependency.
with open('README.md') as f:
    long_description = f.read()

setup(name='FastMatchedFilter',
      version='1.4.0',
      description='Fast time-domain normalised cross-correlation for '
                  'CPU & GPU',
      long_description=long_description,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: GPL License',
        'Programming Language :: Python :: 2.7, 3.5, 3.6, 3.7, 3.8',
        'Topic :: Seismology :: Matched Filter',
      ],
      url='https://github.com/beridel/fast_matched_filter',
      author='William Frank, Eric Beauce',
      author_email='',
      license='GPL',
      packages=['fast_matched_filter'],
      install_requires=['numpy'],
      tests_require=['pytest>=2.0.0'],
      include_package_data=True,
      zip_safe=False,
      cmdclass={
          'build_ext': FastMatchedFilterBuild},
      ext_modules=[FMFExtension('fast_matched_filter.lib.matched_filter_CPU'),
                   FMFExtension('fast_matched_filter.lib.matched_filter_GPU')])
