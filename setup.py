from setuptools import find_packages, setup

setup(
    name="BasisOpt",
    version='1.2',
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
