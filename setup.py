import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agnpy",
    version="0.0.9",
    author="Cosimo Nigro",
    author_email="cosimonigro2@gmail.com.com",
    description="Modelling jetted Active Galactic Nuclei radiative processes with python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cosimoNigro/agnpy",
    packages=setuptools.find_packages(),
    package_data={
        "agnpy": [
            "data/mwl_seds/*.ecsv",
            "data/sampled_seds/*.txt",
            "data/sampled_taus/*.txt",
            "data/ebl_models/*.fits.gz",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["astropy", "numpy", "scipy", "matplotlib"],
    python_requires=">=3.6",
)
