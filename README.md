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

| Model Name | Test Dataset Name | WER | CER |
|------------|--------------|-----|----|
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 | Miami eng herring1 | 87.77% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 | Bark TTS Synthetic 20240605 | 27.50% | 12.57% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + RedPajamaV2 KenLM | Bark TTS Synthetic 20240605 filtered | 29.37% | 16.71% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + synthetic KenLM | Bark TTS Synthetic 20240605 filtered | 24.02% | 14.04% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + 1 redpajama mix 2 synthetic KenLM | Bark TTS Synthetic 20240605 filtered | 24.53% | 14.35% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + 1 redpajama mix 10 synthetic KenLM | Bark TTS Synthetic 20240605 filtered | 24.25% | 14.24% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + 1 redpajama mix 500 synthetic KenLM | Bark TTS Synthetic 20240605 filtered | 24.74% | 14.65% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + 0.01 redpajama mix 0.99 synthetic KenLM | Bark TTS Synthetic 20240605 filtered | 24.61% | 14.49% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + 0.001 redpajama mix 0.999 synthetic KenLM | Bark TTS Synthetic 20240605 filtered | 24.77% | 14.64% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 + synthetic KenLM | Commonvoice es en dev | 15.65% | 8.92% |
| Nemo stt_enes_conformer_transducer_large_codesw beam width 16 | Commonvoice es en dev | 12.38% | 4.85% |
