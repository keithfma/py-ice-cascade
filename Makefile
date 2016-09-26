# Makefile for Sphinx documentation

SPHINXOPTS    =
BUILDDIR      = docs
DOCSOURCEDIR  = docs_src
SOURCEDIR     = py_ice_cascade
TESTDIRS      = `find $(SOURCEDIR) -type d -name tests`

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  help   display this message"
	@echo "  html   (re)generate markup files from source code and build as HTML"
	@echo "  clean  delete everything in the build directory ($(BUILDDIR))"

.PHONY: html
html:
	sphinx-apidoc --private -f -o $(DOCSOURCEDIR) $(SOURCEDIR) $(TESTDIRS)
	sphinx-build -b html $(SPHINXOPTS) $(DOCSOURCEDIR) $(BUILDDIR)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
