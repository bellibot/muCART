from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'muCART',
  packages = ['muCART'],
  version = 'v1.0.2', 
  license='MIT',
  description = 'Measure Inducing Classification and Regression Trees',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Edoardo Belli',
  author_email = 'iedobelli@gmail.com',
  url = 'https://github.com/bellibot/muCART',
  download_url = 'https://github.com/bellibot/muCART/archive/refs/tags/v1.0.1.tar.gz',
  keywords = ['Decision Tree', 'Functional Data', 'Classification', 'Regression'],
  python_requires='>=3.5',
  install_requires=[
          'numpy',
          'matplotlib',
          'Pyomo',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',   # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
  ],
)
