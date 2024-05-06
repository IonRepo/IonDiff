import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cibran",
    version="0.0.1",
    author="Cibrán López Álvarez",
    author_email="cibran.lopez@upc.edu",
    description="Identification and analysis of ion-hopping events in solid state electrolytes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IonRepo/IonDiff",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)