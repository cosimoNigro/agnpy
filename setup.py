import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agnpy",
    version="0.4.0",
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
    install_requires=["astropy>=5.0,<6.0", "numpy>=1.21", "scipy>=1.5,<1.10", "pyyaml", "matplotlib>=3.4", "sherpa", "pre-commit", "gammapy<1.2"],
    python_requires=">=3.9,<3.12",
)
