# py\_ice\_cascade
Python implementation of ICE-CASCADE landscape evolution model 

## Developer Guidelines

### Project Metadata
Metadata (e.g. version, author, etc) is defined in *py_ice_cascade/__init__.py*
and nowhere else.

### Branching 
This project uses a [feature branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow),
meaning that the new feature development should be done in a new branch with an
informative name, and merged with the *master* branch only when it is
completed.   

### Versioning 
This project uses semantic versioning as recommended [here](https://packaging.python.org/distributing/#choosing-a-versioning-scheme)
and [here](http://semver.org/). In short, version numbers are of the form
MAJOR.MINOR.PATCH, where incrementing the MAJOR version indicates incompatible
API changes, the MINOR version indicates added functionality that maintains
compatibility, and PATCH version indicates bug fixes. 
 
### Documentation 
The project documentation is automagically generated from docstrings using
*Sphinx* with the *Napolean* pre-processing extension. Try to stick to the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments)
when writing docstrings. This format is understood by the documentation build
process, and will produce readable documents directly from the code. The
initial *Sphinx* setup followed this [guide](http://gisellezeno.com/tutorials/sphinx-for-python-documentation.html),
with additional modifications of conf.py and Makefile.  To (re)generate the
documentation, you must first run *sphinx-apidoc* to translate formated
docstrings into *rst* files (in [reStructuredText](http://www.sphinx-doc.org/en/stable/rest.html)
markup), then translate these *.rst* files into your prefered format. This
process is mostly automated using the Makefile in *docs/* like so: 
```shell 
cd docs
make apidoc
make html
``` 
The *.rst* files are created in *docs/source* and the output docs are created
in *docs/build/html*. Both should be included in the repository. Keeping the
html documentation in the repository allows them to be published to the web
(on Github) with the following command:
```shell
git subtree push --prefix docs/build/html origin gh-pages
```

### Packaging
This project uses [setuptools](https://pypi.python.org/pypi/setuptools) 
to manage the packaging, build, and installation process. The basic structure 
is provided in this [tutorial](http://python-packaging.readthedocs.io/en/latest/index.html) 
and this [guide](https://packaging.python.org/).

### Installation
Developers may find it useful to install this package while still 
being able to edit the source. See [here](https://packaging.python.org/installing/#installing-from-a-local-src-tree) 
for instructions.

### Testing 
Use the standard library [unittest](https://docs.python.org/2.7/library/unittest.html) 
to create tests for all .py files. The naming convention is to put tests for
*xxx.py* in a file *test_xxx.py* in the same directory. Tests can be run from
the top level or the project with the command: 
```shell
python -m unittest discover -v`
```

### Data 
Testing data for this package is included in the *py_ice_cascade/data* folder.
