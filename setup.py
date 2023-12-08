from setuptools import setup, find_packages

setup(
	name = "BasisOpt",
	version = '1.0.1',
	packages = find_packages(),
	install_requres = [
		'numpy',
		'colorlog',
		'scipy',
		'pandas',
		'monty',
		'basis-set-exchange',
		'mendeleev',
		'matplotlib']
)
