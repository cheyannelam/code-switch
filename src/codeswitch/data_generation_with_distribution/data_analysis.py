import json

import pandas as pd


def read_output():
    outputs = []
    total = 0
    with open("output_json.json", "r", encoding="utf-8") as file:
        for line in file:
            obj = json.loads(line)
            if "text:" in obj:
                total += 1
                obj["text"] = obj.pop("text:")
            outputs.append(obj)
        print("total", total)
    return outputs


def main():
    outputs = read_output()
    df_output = pd.DataFrame(outputs)
    print(df_output.describe())
    print(df_output["syn_cat"].unique())
    print(df_output["syn_cat"].value_counts())
    # print(df_output.head(10)


if __name__ == "__main__":
    main()
