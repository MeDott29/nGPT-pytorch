from setuptools import setup, find_packages

setup(
    name="Data-Generation-and-Testing",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "einops"
    ],
    python_requires=">=3.7",
) 