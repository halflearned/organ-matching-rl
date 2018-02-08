import pandas as pd

df = pd.read_table("~/Documents/kidney/matching/results/confusion_matrices.txt",
                   sep=" ",
                   names=["date", "Environment", "Algorithm", "Additional", "TP", "TN", "FP", "FN"]).dropna()

df = df.drop("date", axis = "columns")

df["Recall (TPR)"] = df["TP"] / (df["TP"] + df["FN"])
df["Specificity (TNR)"] = df["TN"] / (df["TN"] + df["FP"])
df["Precision"] = df["TP"] / (df["TP"] + df["FP"])
df["Accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"])
df["Environment"] = df["Environment"].map({"optn":"OPTN",
                       "abo":"ABO",
                       "saidman":"RSU"})
df["Algorithm"] = df["Algorithm"].map({"lr": "Logistic reg.",
                                       "rf": "Random forests",
                                       "grb": "Gradient boosting"})

df["Additional"] = df["Additional"].map({"both": "Both",
                                         "networkx": "Graph stats",
                                         "node2vec": "Embedding",
                                         "none": "None"})
df[["TP", "TN", "FP", "FN"]] = df[["TP", "TN", "FP", "FN"]].astype(int)

df = df.drop(["TP", "TN", "FP", "FN"], axis = 1)

df = df.sort_values(["Environment", "Algorithm", "Additional"])

df = df.set_index("Environment")

df.to_latex("phd_thesis/tables/traditional_ml.tex",
            escape=False,
            float_format= lambda x: "{:2.3f}".format(x))