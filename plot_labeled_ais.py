import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GEAR_COLOR = {
    "no_fishing": "blue",
    "Trål": "red",
    "Krokredskap": "orange",
    "Snurrevad": "green",
    "Not": "pink",
    "Garn": "brown",
}

print(GEAR_COLOR["Trål"])

df = pd.read_csv(
    "ais_with_ers_labels_01.csv",
    usecols=["mmsi", "date_time_utc", "lon", "lat", "label"],
    low_memory=False
)
df = df.fillna(value={"label": "no_fishing"})
print(df["label"].unique())
print(df.head())

for mmsi, d in df.groupby("mmsi"):
    fig, ax = plt.subplots(figsize=(6, 4))

    d = d.copy()
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values("date_time_utc")

    # New segment every time label changes
    d["segment"] = (d["label"] != d["label"].shift()).cumsum()

    for _, seg_d in d.groupby("segment"):
        gear = seg_d["label"].iloc[0]
        ax.plot(
            seg_d["lon"],
            seg_d["lat"],
            linewidth=1,
            color=GEAR_COLOR[gear]
        )

    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=gear)
        for gear, color in GEAR_COLOR.items()
    ]

    ax.legend(handles=legend_elements, title="Gear")

    plt.show()