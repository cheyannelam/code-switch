# Train

## Training KenLM
A training script can be found at `scripts/train_kemlm.sh`. 

Inside you should configure the location of the training data file.
An example of the file can be found at `/home/public/data/synthetic/20240619/utterance_train.json` on the dialpad mds capstone server.

You should see a trained `kenlm.bin` file after running the script.

## Interpolating KenLM
Once we have two KenLMs, they can be interpolated by the `scripts/mix_kenlm.sh` script.

Simply supply the appropriate `.bin.tmp.arpa` files generated during the KenLM training, and specify the mixing ratio.

The ratio should be tuned according to a dataset that accurately reflects the density of code-switch sentences in production.