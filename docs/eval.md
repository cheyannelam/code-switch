# Evaluation

## Evaluating End to End ASR-LM pipeline WER
Configure the script at `scripts/eval_nemo.sh` and run it. This will give you the WER and CER of a nemo model shallow fused with a KenLM.
### Some notable arguments
- nemo_model_file: Name of nemo model to test. This only works with transducer models. See [here](https://catalog.ngc.nvidia.com/models?filters=application%7CAutomatic+Speech+Recognition%7Cuscs_automatic_speech_recognition%2Cplatform%7CNeMo%7Cpltfm_nemo&orderBy=weightPopularDESC&query=&page=&pageSize=) for some potentially compatible models.
- input_manifest: Path to a test data file.
- kenlm_model_file: Path to a trained KenLM model file.
- preds_output_folder: Path to a folder for saving model predictions.

