SAVE_ROOT='/home/public/data/synthetic/example'
NUM_SAMPLES=100

mkdir -p $SAVE_ROOT
python -m codeswitch.data_generation_with_distribution.domain_extraction
python -m codeswitch.data_generation_with_distribution.data_generation_pos_distribution --num-utterance=$NUM_SAMPLES  --workers=32
jq -r '.text' output.json > utterance.txt
awk '{if(rand()<0.9) {print $0 > "utterance_train.txt"} else {print $0 > "utterance_test.txt"}}' utterance.txt
mv utterance.txt $SAVE_ROOT/utterance.txt
mv utterance_train.txt $SAVE_ROOT/utterance_train.txt
mv output.json $SAVE_ROOT/utterance.json
python -m codeswitch.text_to_speech.tts_generation --data-path=utterance_test.txt  --output-path=$SAVE_ROOT --use-gpu=False
python -m codeswitch.evaluation.whisper_generate_transcripts --data-path=$SAVE_ROOT/tts_manifest.json  --output-foldername=transcripts
python -m codeswitch.evaluation.wer_evaluation --data-path=transcripts/transcripts.json --groundtruth-path=$SAVE_ROOT/tts_manifest.json --output-stats-path=wer.tsv --output-manifest-path=$SAVE_ROOT/manifest_test.json