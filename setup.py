from setuptools import setup, find_packages

setup(name = 'shrec',
      packages=['shrec'],
      version='0.1',
      # packages=find_packages(),
      # install_requires = ["h5py", "numpy", "scipy", "pandas", "scanpy", "graspologic"],
      # extras_require = {
      #   "scikit-learn": ["scikit-learn"],
      #   "matplotlib": ["matplotlib"],
      #   "seaborn": ["seaborn"]
      # },
      package_dir={'shrec': 'shrec'},
      package_data={'shrec': ['data/*']},
     )
