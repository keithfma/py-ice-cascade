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
	entry_points={'console_scripts' : [
		'ice-cascade = py_ice_cascade.ice_cascade:cli', 
        'ice-cascade-create-example = py_ice_cascade.create_example:main']}
)
