import os
from setuptools import setup, find_packages

# exec (open('version.py').read())

setup(name='segtools',
      version='0.0.1',
      description='Segmentation and tracking scores, graphs and recoloring using networkx and numpy',
      # url='https://github.com/maweigert/spimagine',
      author='Coleman Broaddus',
      author_email='broaddus@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=['segtools'],
      # packages=find_packages(),
      # zip_safe=False)
      )
