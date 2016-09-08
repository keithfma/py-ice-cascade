# Makefile for Sphinx documentation

SPHINXOPTS    =
BUILDDIR      = docs
SOURCEDIR     = docs_src

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  help   display this message"
	@echo "  html   (re)generate markup files from source code and build as HTML"
	@echo "  clean  delete everything in the build directory ($(BUILDDIR))"

.PHONY: html
html:
	sphinx-apidoc --private -f -o $(SOURCEDIR) ./py_ice_cascade ./py_ice_cascade/test_*
	sphinx-build -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*
