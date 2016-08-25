from setuptools import setup
import py_ice_cascade as ic

setup(
	name=ic.__name__,
	version=ic.__version__,
	description=ic._description,
	url=ic._url,
	author=ic._author, 
	author_email=ic._author_email, 
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Programming Language :: Python :: 3.5',
		'Operating System :: POSIX :: Linux',
		'License :: OSI Approved :: MIT License'],
	packages=['py_ice_cascade'],
	package_data={'py_ice_cascade' : ['data/*']},
	install_requires=[],
)
