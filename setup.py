from setuptools import find_packages, setup

setup(
    name="BasisOpt",
    version='1.1',
    packages=find_packages(),
    install_requres=[
        'numpy',
        'colorlog',
        'scipy',
        'pandas',
        'monty',
        'basis-set-exchange',
        'mendeleev',
        'matplotlib',
        'ray',
    ],
)
