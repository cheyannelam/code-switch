python -m codeswitch.train_kenlm.eval_beamsearch_ngram_transducer nemo_model_file="stt_enes_conformer_transducer_large_codesw" \
       input_manifest="/home/public/data/Miami/manifests/eng/herring1.json" \
       kenlm_model_file="kenlm.bin" \
       beam_width="[16]" \
       beam_alpha="[1.0]" \
       preds_output_folder="predictions.txt" \
       probs_cache_file=null \
       decoding_strategy="maes" \
       use_amp="true" \
       num_workers=1 \
       beam_batch_size=128 \
       acoustic_batch_size=128 \
       device="cpu"