import os
from setuptools import setup, find_packages

# exec (open('version.py').read())

setup(name='detsegtra',
      version='0.0.1alpha',
      url='https://github.com/mpicbg-csbd/detsegtra',
      description='tools for detection, segmentation, tracking of nuclei in 3D fluorescence microsopy data',
      author='Coleman Broaddus',
      author_email='broaddus@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=find_packages(),
      py_modules=['segtools'],
      install_requires=[
         'numpy',
         'matplotlib',
         'tifffile',
         'pandas',
         'pulp',
         # 'tensorflow',
         # 'keras',
         'torch',
         'torchsummary',
         'scikit-image',
         'pykdtree',
         'termcolor',
         'sortedcollections',
         'tabulate', 
         'scikit-learn',
         'seaborn',
         'networkx',
         'numba',

      ],
      # zip_safe=False)
      )
