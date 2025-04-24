#!/bin/bash
sphinx-apidoc --ext-autodoc -o doc ./delex/
pushd doc
make html
