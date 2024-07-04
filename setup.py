from distutils.core import setup

with open('docs/index.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
  name = 'IonDiff',
  packages = ['IonDiff'],
  version = '1.2',
  license=license,
  description = 'Unsupervised identification and analysis of ion-hopping events in solid state electrolytes.',
  long_description=readme,
  author = 'Cibrán López Álvarez',
  author_email = 'cibran.lopez@upc.edu',
  url = 'https://github.com/IonRepo/IonDiff',
  download_url = 'https://github.com/IonRepo/IonDiff/archive/refs/tags/0.1.tar.gz',
  keywords = ['Ionic Diffusion', 'Unsupervised Clustering', 'Machine Learning'],
  install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'multiprocess',
          'seaborn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
