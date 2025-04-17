#!/bin/bash

# Create the bin directory if it doesn't exist
mkdir -p bin

# Run make to build the library
make

# Build the aida_example executable
make aida_example