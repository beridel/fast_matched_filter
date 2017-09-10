"""
Minimal setup file for the fast_matched_filter library for Python packaging
"""

from setuptools import setup


with open('README.md') as f:
    long_description = f.read()

# TODO: Set up compilation of both CPU and GPU libs (if hardware supported)
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
      include_package_data=True,
      zip_safe=False)
