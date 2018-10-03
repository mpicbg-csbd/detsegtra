import os
from setuptools import setup, find_packages

# exec (open('version.py').read())

setup(name='segtools',
      version='0.0.1',
      description='Segmentation and tracking scores, graphs and recoloring using networkx and numpy',
      author='Coleman Broaddus',
      author_email='broaddus@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=find_packages(),
     install_requires=[
         'numpy',
         'matplotlib',
         'pandas',
         'tifffile',
         'pulp',
         'tensorflow',
         'keras',
         'scikit-image',
         'pykdtree',
         'termcolor',
         'sortedcollections',
         'tabulate', 
         'scikit-learn',
         'seaborn',
         'networkx',
         'numba'
      ],      
      # packages=find_packages(),
      # zip_safe=False)
      )
