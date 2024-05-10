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
        "openai",
        "gtts",
        "pandas",
        "python-dotenv",
    ],
)
