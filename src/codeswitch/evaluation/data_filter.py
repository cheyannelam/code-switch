from lingua import Language, LanguageDetectorBuilder

from codeswitch.dataloader import (  # noqa
    # check_device,
    read_splitted_miami,
    # read_synthetic_data,
)


def extract_codeswitch_sentences(sentences, cs_coef=0.9):
    languages = [Language.ENGLISH, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    cs_sentences = []
    cs_index = []
    for i, sentence in enumerate(sentences):
        # print(sentence)

        if len(sentence) < 20:  # filter away meaningless sentences such as 'hmmm'
            continue

        confidence_values = detector.compute_language_confidence_values(sentence)
        largest_confidence = (
            confidence_values[0].value
            if confidence_values[0].value > confidence_values[1].value
            else confidence_values[1].value
        )
        if 0 < largest_confidence < cs_coef:
            print(confidence_values[0].value, confidence_values[1].value, sentence)
            cs_sentences.append(sentence)
            cs_index.append(i)
    return cs_sentences, cs_index


def main():

    data = read_splitted_miami()
    # data = read_synthetic_data()
    sentences = [text for _, text in data]
    cs_sentences = extract_codeswitch_sentences(sentences, 0.8)

    print("cs len:  ", len(cs_sentences))
    print("data len:", len(data))

    for sent in cs_sentences:
        print(sent)


if __name__ == "__main__":
    main()
