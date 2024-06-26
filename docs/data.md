# Data
Several datasets are available on the Dialpad mds capstone server at `/home/public/data`.

## Synthetic Data

The newest version of synthetic data can be found at `/home/public/data/synthetic/20240619`.

It contains 50000 generated sentences, and about 1000 synthetic audio for the test set using Bark TTS.
```
/home/public/data/synthetic/20240619
|-- ttsaudio/            # Synthetic codeswitch audio for ASR pipeline tests
|-- manifest_test.json   # Test data format for Nemo evalution scripts
|-- top_topics.txt       # Topics extracted from the provided dialpad data
|-- utterance.json       # All the synthetic codeswitch text data
|-- utterance_train.json # 90% of the synthetic codeswitch text data for training
`-- utterance_train.txt  # Simplified train data format for KenLM
```

## Commonvoice
The folder contains the English and Spanish Text-Audio data, and also text-only data extracted by Dialpad.

If the text data are used to train an ASR LM, you might get poor results when evaluated on production data, because the text transcript of CommonVoice are fill-in-the-blanks style template generated. The problem is especially pronounced for low resource languages.

## Dialpad
This dataset is provided by Dialpad, and contains anonymized synthetic ASR transcripts from production audio recordings.

It contains ASR errors so they are not directly applicable for LM training. 

## Miami
This contains the code-switch dataset from `http://bangortalk.org.uk/speakers.php?c=miami`.

The quality of the data is rather poor and not used for the final product.

## Oscar
Oscar is a cleaned and deduped general text corpus of CommonCrawl. This can be used for training general purpose language models. However the data are not conversation transcripts so there would be a domain mismatch for ASR LMs.

## Redpajama V2
RedpajamaV2 is another cleaned and deduped general text corpus of CommonCrawl. It is commonly used for training open source LLMs.

However, as with Oscar, the data are not conversation transcripts so there would be a domain mismatch for ASR LMs.
