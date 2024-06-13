# code-switch

## Setup
1. Install external dependencies if your system do no have them.
2. Create a conda environment and activate it
3. Install python 3.11 in your conda environment.
4. Run `make`

### External dependencies:
  - libboost

## Run linters
Run `make lint`

## Run formatters
Run `make lint-format`

### Results:

| Model Name | Dataset Name | WER |
|------------|--------------|-----|
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 | Miami eng herring1 | 87.77% |
