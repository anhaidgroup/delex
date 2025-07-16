#!/bin/bash
cd "$(dirname "$0")"
make clean
make html
echo "Documentation built successfully!"
echo "Open _build/html/index.html in your browser to view the documentation."
