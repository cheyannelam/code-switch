import pickle  # nosec
from pathlib import Path

import click
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from transformers import AutoTokenizer


@click.command()
@click.option(
    "--data-root",
    default="./data/commonvoice",
    help="Path to line sepearated text data folder",
)
@click.option(
    "--ngram-order", default=3, help="Max ngram order for the language model", type=int
)
def main(data_root, ngram_order):
    # Mixtral tokenizer is multilingual on English, French and Spanish
    # Not using the official one to circumvent the authentication issue
    tokenizer = AutoTokenizer.from_pretrained(
        "cognitivecomputations/dolphin-2.7-mixtral-8x7b"
    )

    data_fnames = {
        "es": "cv-corpus-10.0-train_es.txt",
        "en": "cv-corpus-17.0-train_en.txt",
    }

    data = []
    vocab = set()
    for fname in data_fnames.values():
        data_fpath = Path(data_root) / fname
        lang_data = data_fpath.read_text()
        tokens = [tokenizer.tokenize(x) for x in lang_data.split("\n") if x]
        train, lang_vocab = padded_everygram_pipeline(ngram_order, tokens)
        data += train
        vocab.update(lang_vocab)

    lm = MLE(ngram_order)
    lm.fit(data, vocab)

    with open(f"{ngram_order}gram-lm.pkl", "wb") as f:
        pickle.dump(lm, f)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
