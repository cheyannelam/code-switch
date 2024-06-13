python -m codeswitch.train_kenlm.train_kenlm nemo_model_file="stt_enes_conformer_transducer_large_codesw" \
                          train_paths="[/home/public/data/synthetic.txt]"\
                          kenlm_bin_path="../build/bin" \
                          kenlm_model_file="kenlm.bin" \
                          ngram_length=3 \
                          preserve_arpa=true