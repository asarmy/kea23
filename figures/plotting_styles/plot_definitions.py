"""This file defines the color codes and line styles used in the report plots."""

colors = {
    "lightblue": "#0096ff",
    "darkblue": "#00008b",
    "lightred": "#fc46aa",
    "darkred": "#d10000",
    "orange": "#ff5f1f",
    "purple": "#800080",
    "lightgreen": "#4cbb17",
    "darkgreen": "#008000",
    "brown": "#6f4e37",
    "gray": "#818589",
    "black": "#000000",
}

model_colors = {
    "MR11": colors.get("lightblue"),
    "PEA11": colors.get("lightred"),
    "WC94-all": colors.get("purple"),
    "WC94-ss": colors.get("lightgreen"),
    "WC94-rv": colors.get("lightgreen"),
    "WC94-nm": colors.get("lightgreen"),
    "KEA23": colors.get("black"),
}

model_linestyles = {
    "MR11": (0, (6, 2)),
    "PEA11": (0, (6, 2)),
    "WC94-all": (0, (3, 1)),
    "WC94-ss": "dashdot",
    "WC94-rv": "dashdot",
    "WC94-nm": "dashdot",
    "KEA23": "solid",
}

model_labels = {
    "MR11": "MR11",
    "PEA11": "PEA11",
    "WC94-all": "WC94 (All)",
    "WC94-ss": "WC94 (SS)",
    "WC94-rv": "WC94 (RV)",
    "WC94-nm": "WC94 (NM)",
    "KEA23": "KEA23",
}
