import json
import os

import pandas as pd
import torch


def check_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def read_splitted_miami():
    """return (audio_filepath_list, text_list)"""
    with open(
        "/home/public/data/Miami/manifests/eng/herring1.json", "r", encoding="utf-8"
    ) as file:
        lines = file.read().splitlines()
    dicts = [json.loads(line) for line in lines]
    return [(d["audio_filepath"], d["text"]) for d in dicts]


def read_synthetic_data(data_path=""):
    """return (audio_filepath_list, text_list)"""
    if data_path == "":
        data_path = os.path.join(
            os.path.dirname(__file__), "../../../data/synthetic_code_switch_data"
        )
        data_path = os.path.normpath(data_path)

    val_dataset = pd.read_csv(os.path.join(data_path, "output.csv"))
    val_dataset["audio_filename"] = val_dataset["audio_filename"].apply(
        lambda x: data_path + "/audio/" + x
    )
    return list(
        zip(
            val_dataset["audio_filename"].tolist(),
            val_dataset["code-switched"].tolist(),
        )
    )


def read_text_data(data_path=None):
    """return a text list"""
    if data_path is None:
        data_path = "/home/public/data/utterance.txt"
        print("No specialized data_path, reading default utterance.txt dataset")
    with open(data_path, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
    return lines


def read_transcript_pairs(data_path):
    with open(data_path, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
        lines = [line.split("\t") for line in lines]
    return lines


def read_transcript_tsv(data_path):
    with open(data_path, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
        lines = [line.split("\t")[0] for line in lines]
    return lines
