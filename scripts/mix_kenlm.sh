python -m codeswitch.train_kenlm.ngram_merge \
  --ngram_bin_path "" \
  --kenlm_bin_path "../build/bin" \
  --arpa_a kenlm_redpajama.bin.tmp.arpa \
  --alpha 1 \
  --arpa_b kenlm_synthetic.bin.tmp.arpa \
  --beta 10 \
  --out_path mixed