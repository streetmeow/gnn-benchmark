SGC_DICT = {

    ("texas", "sgc"): {
        "layer": [1],          # placeholder (SGC has no depth)
        "hidden": [16],
        "lr": [0.01],
        "dropout": [0.0],
        "wd": [0],
        # "k_value": [1]
    },

    ("cornell", "sgc"): {
        "layer": [1],
        "hidden": [16],
        "lr": [0.01],
        "dropout": [0.0],
        "wd": [0],
        # "k_value": [1]
    },

    ("cora", "sgc"): {
        "layer": [1],
        "hidden": [32],
        "lr": [0.01],
        "dropout": [0.0],
        "wd": [0],
        # "k_value": [2]
    },

    ("citeseer", "sgc"): {
        "layer": [1],
        "hidden": [32],
        "lr": [0.01],
        "dropout": [0.0],
        "wd": [0],
        # "k_value": [2]
    },

    ("pubmed", "sgc"): {
        "layer": [1],
        "hidden": [64],
        "lr": [0.005],
        "dropout": [0.0],
        "wd": [0],
        # "k_value": [3]
    },

}
