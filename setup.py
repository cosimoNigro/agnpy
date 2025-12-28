import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agnpy",
    version="0.5.0",
    author="Cosimo Nigro",
    author_email="cosimonigro2@gmail.com.com",
    description="Modelling jetted Active Galactic Nuclei radiative processes with python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cosimoNigro/agnpy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=["astropy>=7.0", "numpy>=1.24,<2.3", "pandas>=1.5", "scipy>=1.14", "pyyaml>=6.0", "matplotlib>=3.9.1", "sherpa", "pre-commit", "gammapy>=1.3"],
    python_requires=">=3.11,<3.14",
)
