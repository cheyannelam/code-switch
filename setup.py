import setuptools  # isort:skip # pylint: disable=import-error


setuptools.setup(
    name="codeswitch",
    author="",
    author_email="",
    python_requires=">=3.10",
    install_requires=[
        "icecream>=2.1.3",
        "tqdm>=4.65.0",
        "flake8>=6.0.0",
        "black>=23.3.0",
        "isort>=5.12.0",
        "autoflake>=2.2.0",
        "pylint>=2.17.5",
        "yq",
        "openai",
        "gtts",
        "pandas",
        "python-dotenv",
        "yq",
        "sentencepiece",
        "nltk",
        "nemo_toolkit[asr]",
        "pylangacq",
        "pydub",
        "kenlm",
        "youtokentome @ git+https://github.com/gburlet/YouTokenToMe.git@dependencies",
        "sox<=1.4.0",  # sox 1.5.0 build time dependency is broken
    ],
)
