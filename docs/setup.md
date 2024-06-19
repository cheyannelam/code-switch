# Setup

## General Instructions:
1. Install external dependencies if your system do no have them.
2. Create a conda environment and activate it
3. Install python 3.11 in your conda environment.
4. Run `make`

## A non-exhausive list of external dependencies
Depending on the OS, you may need extra dependencies from your OS package manager.. These are some dependencies required on a Debian 11.
- libboost
- libsox
- cmake

## Docker Environment
If the installation does not work for some reason, you can refer to the github action setup at `.github/workflows/lint.yml` for an example of a functioning docker container.