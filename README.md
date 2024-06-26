# code-switch
Detailed documentations can be found at `/docs`



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

| Model Name | Test Dataset Name | WER | CER |
|------------|--------------|-----|----|
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 | Miami eng herring1 | 87.77% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + RedPajamaV2 KenLM | Bark TTS Synthetic 20240605 | 29.37% | 16.71% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + synthetic KenLM | Bark TTS Synthetic 20240605 | 24.02% | 14.04% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + synthetic KenLM | Commonvoice es en dev | 15.65% | 8.92% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + redpajama KenLM| Commonvoice es en dev | 13.62% | 7.65% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + Commonvoice KenLM| Commonvoice es en dev | 13.62% | 7.65% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + synthetic 50k KenLM | Bark TTS Synthetic 20240605 | 25.34% | 14.76% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 | Bark TTS Synthetic 20240619 | 29.78% | 13.88% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 | Commonvoice es en dev | 12.38% | 4.85% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + synthetic 50k KenLM| Bark TTS Synthetic 20240619 | 26.63% | 17.09% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + synthetic 50k KenLM| Commonvoice es en dev | 15.67 % | 8.82% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + commonvoice lm mix synthetic 50k KenLM| Bark TTS Synthetic 20240619 | 27.70% | 18.06% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + commonvoice lm mix synthetic 50k KenLM| Commonvoice es en dev | 14.28% | 8.05% |