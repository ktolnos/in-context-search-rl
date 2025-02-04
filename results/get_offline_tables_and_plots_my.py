import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rliable import library as rly, metrics, plot_utils
from plot_config import datasets_in_use, groups_to_plot, k_to_plot, list_of_list

dataframe = pd.read_csv("runs_tables/offline_urls.csv")
with open("bin/offline_scores.pickle", "rb") as handle:
    full_scores = pickle.load(handle)

with open("bin/myscores.pkl", "rb") as handle:
    my_scores = pickle.load(handle)

with open("bin/myscores_100.pkl", "rb") as handle:
    my_scores_100 = pickle.load(handle)

for algo in my_scores:
    full_scores[algo] = my_scores[algo]

for algo in my_scores_100:
    full_scores[algo] = my_scores_100[algo]

os.makedirs("./out", exist_ok=True)


def get_average_scores(scores):
    avg_scores = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    stds = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    for algo in scores:
        for data in scores[algo]:
            sc = scores[algo][data]
            if len(sc) > 0:
                ml = min(map(len, sc))
                sc = [s[:ml] for s in sc]
                scores[algo][data] = sc
                avg_scores[algo][data] = np.mean(sc, axis=0)
                stds[algo][data] = np.std(sc, axis=0)

    return avg_scores, stds


def get_max_scores(scores):
    avg_scores = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    stds = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    for algo in scores:
        for data in scores[algo]:
            sc = scores[algo][data]
            if len(sc) > 0:
                ml = min(map(len, sc))
                sc = [s[:ml] for s in sc]
                scores[algo][data] = sc
                max_scores = np.max(sc, axis=1)
                avg_scores[algo][data] = np.mean(max_scores)
                stds[algo][data] = np.std(max_scores)

    return avg_scores, stds


def get_last_scores(avg_scores, avg_stds):
    last_scores = {
        algo: {
            ds: avg_scores[algo][ds][-1] if avg_scores[algo][ds] is not None else None
            for ds in avg_scores[algo]
        }
        for algo in avg_scores
    }
    stds = {
        algo: {
            ds: avg_stds[algo][ds][-1] if avg_stds[algo][ds] is not None else None
            for ds in avg_scores[algo]
        }
        for algo in avg_scores
    }
    return last_scores, stds


avg_scores, avg_stds = get_average_scores(full_scores)
max_scores, max_stds = get_max_scores(full_scores)
last_scores, last_stds = get_last_scores(avg_scores, avg_stds)
# Dict[algo, Dict[dataset, List[scores]]]

def add_domains_avg(scores):
    for algo in scores:
        locomotion = [
            scores[algo][data]
            for data in datasets_in_use.intersection([
                "halfcheetah-medium-v2",
                "halfcheetah-medium-replay-v2",
                "halfcheetah-medium-expert-v2",
                "hopper-medium-v2",
                "hopper-medium-replay-v2",
                "hopper-medium-expert-v2",
                "walker2d-medium-v2",
                "walker2d-medium-replay-v2",
                "walker2d-medium-expert-v2",
            ])
        ]
        antmaze = [
            scores[algo][data]
            for data in datasets_in_use.intersection([
                "antmaze-umaze-v2",
                "antmaze-umaze-diverse-v2",
                "antmaze-medium-play-v2",
                "antmaze-medium-diverse-v2",
                "antmaze-large-play-v2",
                "antmaze-large-diverse-v2",
            ])
        ]
        maze2d = [
            scores[algo][data]
            for data in datasets_in_use.intersection([
                "maze2d-umaze-v1",
                "maze2d-medium-v1",
                "maze2d-large-v1",
            ])
        ]

        adroit = [
            scores[algo][data]
            for data in datasets_in_use.intersection([
                "pen-human-v1",
                "pen-cloned-v1",
                "pen-expert-v1",
                "door-human-v1",
                "door-cloned-v1",
                "door-expert-v1",
                "hammer-human-v1",
                "hammer-cloned-v1",
                "hammer-expert-v1",
                "relocate-human-v1",
                "relocate-cloned-v1",
                "relocate-expert-v1",
            ])
        ]

        scores[algo]["locomotion avg"] = np.mean(locomotion)
        scores[algo]["antmaze avg"] = np.mean(antmaze)
        scores[algo]["maze2d avg"] = np.mean(maze2d)
        scores[algo]["adroit avg"] = np.mean(adroit)

        scores[algo]["total avg"] = np.mean(
            np.hstack((locomotion, antmaze, maze2d, adroit))
        )


add_domains_avg(last_scores)
add_domains_avg(max_scores)

algorithms = [
    "BC",
    "10% BC",
    "TD3+BC",
    "AWAC",
    "CQL",
    "IQL",
    "ReBRAC",
    "SAC-N",
    "EDAC",
    "DT",
] + [g+k for g in groups_to_plot for k in map(str, k_to_plot)] + [gr + "_inv_" +k for gr in groups_to_plot for k in map(str, k_to_plot)]
datasets = datasets_in_use #dataframe["dataset"].unique()



ordered_datasets = datasets_in_use.intersection([
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    "locomotion avg",
    "maze2d-umaze-v1",
    "maze2d-medium-v1",
    "maze2d-large-v1",
    "maze2d avg",
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
    "antmaze avg",
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
    "door-human-v1",
    "door-cloned-v1",
    "door-expert-v1",
    "hammer-human-v1",
    "hammer-cloned-v1",
    "hammer-expert-v1",
    "relocate-human-v1",
    "relocate-cloned-v1",
    "relocate-expert-v1",
    "adroit avg",
    "total avg",
])

"""# Tables"""


def get_table(
    scores,
    stds,
    pm="$\\pm$",
    delim=" & ",
    row_delim="\\midrule",
    row_end=" \\\\",
    row_begin="",
    add_header_delim=False,
):
    rows = [row_begin + delim.join(["Task Name"] + algorithms) + row_end]
    if add_header_delim:
        rows.append(row_begin + "|".join(["---"] * (len(algorithms) + 1)) + row_end)
    prev_env = "halfcheetah"
    for data in ordered_datasets:
        env = data.split("-")[0]
        if env != prev_env:
            if len(row_delim) > 0:
                rows.append(row_delim)
            prev_env = env

        row = [data]

        for algo in algorithms:
            if data in stds[algo]:
                row.append(f"{scores[algo][data]:.2f} {pm} {stds[algo][data]:.2f}")
            else:
                row.append(f"{scores[algo][data]:.2f}")
        rows.append(row_begin + delim.join(row) + row_end)
    return "\n".join(rows)

os.makedirs("out", exist_ok=True)

print(get_table(last_scores, last_stds))
print("\n")
print(get_table(max_scores, max_stds))
print("\n")

with open("out/tables.md", "w") as f:
    f.write("# Last Scores\n")
    f.write(get_table(last_scores, last_stds, "±", "|", "", "|", "|", True))
    f.write("\n\n\n# Max Scores\n")
    f.write(get_table(max_scores, max_stds, "±", "|", "", "|", "|", True))

plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["figure.dpi"] = 300
sns.set(style="ticks", font_scale=1.5)

def convert_dataset_name(name):
    name = name.replace("v2", "")
    name = name.replace("v1", "")
    name = name.replace("v0", "")
    name = name.replace("medium-", "m-")
    name = name.replace("umaze-", "u-")
    name = name.replace("large-", "l-")
    name = name.replace("replay-", "re-")
    name = name.replace("random-", "ra-")
    name = name.replace("expert-", "e-")
    name = name.replace("play-", "p-")
    name = name.replace("diverse-", "d-")
    name = name.replace("human-", "h-")
    name = name.replace("cloned-", "c-")
    return name[:-1]


def plot_bars(scores, save_name):
    agg_l = []
    for algo in algorithms:
        for data in scores[algo]:
            if data not in datasets_in_use:
                continue
            line = convert_dataset_name(data)
            agg_l.append([algo, line, scores[algo][data]])

    df_agg = pd.DataFrame(agg_l, columns=["Algorithm", "Dataset", "Normalized Score"])

    sns.set(style="ticks", font_scale=2)
    plt.rcParams["figure.figsize"] = (20, 10)  # (10, 6)

    b = sns.barplot(
        data=df_agg,
        x="Dataset",
        y="Normalized Score",
        hue="Algorithm",
    )
    plt.grid()
    # plt.tight_layout()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    sns.move_legend(b, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f"out/bars_{save_name}_all.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


plot_bars(last_scores, "last")
plot_bars(max_scores, "max")

