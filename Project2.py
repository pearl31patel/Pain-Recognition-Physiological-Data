import sys
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import random
random.seed(5)


data_type = sys.argv[1]
file_path = sys.argv[2]


def get_features(values):
    values = np.array(values, dtype=float)
    return [np.mean(values), np.var(values), np.min(values), np.max(values)]


def load_data(file):
    data = {}

    with open(file) as f:
        for row in csv.reader(f):
            subject, signal, label = row[0], row[1], row[2]
            values = [float(x) for x in row[3:]]

            if subject not in data:
                data[subject] = {"Pain": {}, "No Pain": {}}

            data[subject][label][signal] = values

    return data


def create_features(records, data_type):
    signals = {
        "dia": "BP Dia_mmHg",
        "sys": "LA Systolic BP_mmHg",
        "eda": "EDA_microsiemens",
        "res": "Respiration Rate_BPM"
    }

    X, y, groups = [], [], []

    for subject in records:
        for label in ["No Pain", "Pain"]:
            if label not in records[subject]:
                continue

            if data_type == "all":
                features = []
                for s in signals.values():
                    features += get_features(records[subject][label][s])
            else:
                features = get_features(records[subject][label][signals[data_type]])

            X.append(features)
            y.append(1 if label == "Pain" else 0)
            groups.append(subject)

    return np.array(X), np.array(y), np.array(groups)

        

def run_model(X, y, groups, data_type):
    model = RandomForestClassifier(random_state=5)
    gkf = GroupKFold(10)

    acc, prec, rec, cms = [], [], [], []

    for train, test in gkf.split(X, y, groups):
        y_pred = model.fit(X[train], y[train]).predict(X[test])

        acc.append(accuracy_score(y[test], y_pred))
        prec.append(precision_score(y[test], y_pred))
        rec.append(recall_score(y[test], y_pred))
        cms.append(confusion_matrix(y[test], y_pred))

    print("\nAccuracy:", np.mean(acc))
    print("Precision:", np.mean(prec))
    print("Recall:", np.mean(rec))
    print("\nConfusion Matrix:\n", np.mean(cms, axis=0))

    filename = f"results/{data_type}.txt"

    with open(filename, "w") as f:
        f.write(f"Accuracy: {np.mean(acc)}\n")
        f.write(f"Precision: {np.mean(prec)}\n")
        f.write(f"Recall: {np.mean(rec)}\n")
        f.write(f"Confusion Matrix:\n{np.mean(cms, axis=0)}\n")

def plot_box(X,data_type):
    plt.figure()
    plt.boxplot(X)
    plt.title("Feature Variability")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.savefig(f"boxplot/{data_type}_boxplot.png")

def plot_signals(data):
    subject = random.choice(list(data.keys()))
    label = random.choice(["Pain", "No Pain"])
    signals = data[subject][label]
    plt.figure()

    for name, values in signals.items():
        plt.plot(values, label=name)

    plt.title("Physiological Signals")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.savefig("signalsplot/signals_plot.png")


data = load_data(file_path)
X, y, groups = create_features(data, data_type)
run_model(X, y, groups, data_type)
plot_box(X,data_type)
plot_signals(data)