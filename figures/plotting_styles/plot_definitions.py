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
    "PEA11-ellip": colors.get("lightred"),
    "PEA11-quad": colors.get("lightred"),
    "PEA11-bilin": colors.get("lightred"),
    "MR11-d_ad": colors.get("lightblue"),
    "MR11-d_md": colors.get("darkblue"),
    "YEA03": colors.get("orange"),
    "YEA03-d_ad": colors.get("orange"),
}

model_linestyles = {
    "MR11": (0, (6, 2)),
    "PEA11": (0, (6, 2)),
    "WC94-all": (0, (3, 1)),
    "WC94-ss": "dashdot",
    "WC94-rv": "dashdot",
    "WC94-nm": "dashdot",
    "KEA23": "solid",
    "PEA11-ellip": (0, (6, 2)),
    "PEA11-quad": (0, (3, 1)),
    "PEA11-bilin": "dashdot",
    "MR11-d_ad": (0, (6, 2)),
    "MR11-d_md": "dashdot",
    "YEA03-d_ad": (0, (3, 1)),
    "YEA03": (0, (3, 1)),
}

model_labels = {
    "MR11": "MR11",
    "PEA11": "PEA11",
    "WC94-all": "WC94 (All)",
    "WC94-ss": "WC94 (SS)",
    "WC94-rv": "WC94 (RV)",
    "WC94-nm": "WC94 (NM)",
    "KEA23": "KEA23",
    "PEA11-ellip": "PEA11 Elliptical",
    "PEA11-quad": "PEA11 Quadratic",
    "PEA11-bilin": "PEA11 Bilinear",
    "MR11-d_ad": "MR11 D/AD",
    "MR11-d_md": "MR11 D/MD",
    "YEA03-d_ad": "YEA03 D/AD",
    "YEA03": "YEA03",
}
