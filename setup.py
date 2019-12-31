from setuptools import setup, find_packages

setup(
    name="agnpy",  # Replace with your own username
    version="0.0.1",
    author="Cosimo Nigro",
    author_email="cosimonigro2@gmail.com",
    description="Modelling jetted Active Galactic Nuclei radiative processes with python",
    url="https://github.com/cosimoNigro/agnpy",
    packages=find_packages("agnpy"),
    package_dir={"": "agnpy"},
    package_data={"agnpy": ["data/*.npz"]},
    install_requires=["numpy", "astropy", "numba", "matplotlib"],
    python_requires=">=3.7",
)
