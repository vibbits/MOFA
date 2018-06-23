from setuptools import setup
from setuptools import find_packages

import sys

def setup_package():
  install_requires = ['pandas', 'scipy', 'numpy', 'sklearn', 'argparse', 'h5py']
  metadata = dict(
      name = 'MOFA',
      version = '1.0',
      description = 'Multi-Omics Factor Analysis',
      url = 'https://github.com/PMBio/MOFA',
      author = 'Ricard Argelaguet, Damien Arnol and Britta Velten',
      author_email = 'ricard.argelaguet@gmail.com',
      license = 'MIT',
      packages = find_packages(),
      install_requires = install_requires
    )

  setup(**metadata)

if __name__ == '__main__':
  if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')
    
  setup_package()