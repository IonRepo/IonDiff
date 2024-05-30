from distutils.core import setup

setup(
  name = 'IonDiff',
  packages = ['IonDiff'],
  version = '0.1',
  license='MIT',
  description = 'Unsupervised identification and analysis of ion-hopping events in solid state electrolytes.',
  author = 'Cibrán López Álvarez',
  author_email = 'cibran.lopez@upc.edu',
  url = 'https://github.com/IonRepo/IonDiff',
  download_url = 'https://github.com/IonRepo/IonDiff/archive/refs/tags/0.1.tar.gz',
  keywords = ['Ionic Diffusion', 'Unsupervised'],
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
