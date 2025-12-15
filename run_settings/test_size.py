HP_TEST_SEARCH_SPACE = {

    # ======================

    # 1) CITESEER

    # ======================

    ("citeseer", "gcn"): {

        "layer": [2],                 # 2-layer 더 안정적

        "hidden": [128],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-4],

    },

    ("citeseer", "gat"): {
        "layer": [2],
        "hidden": [192],
        "lr": [ 0.001],
        "dropout": [ 0.2],
        "wd": [0.001],
    },

    ("citeseer", "gin"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.2],

        "wd": [5e-4],

    },

    ("citeseer", "graphsage"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5],

        "wd": [ 5e-4],

    },



    # ======================

    # 2) CORA

    # ======================

    ("cora", "gcn"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.01],

        "dropout": [0.2],

        "wd": [5e-5,],

    },

    ("cora", "gat"): {
        "layer": [2, ],
        "hidden": [32,],
        "lr": [0.005],
        "dropout": [0.2, ],
        "wd": [0, ],
    },

    ("cora", "gin"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5],

        "wd": [5e-5, ],

    },

    ("cora", "graphsage"): {

        "layer": [3],

        "hidden": [64, ],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-4],

    },



    # ======================

    # 3) PUBMED

    # ======================

    ("pubmed", "gcn"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.01, ],

        "dropout": [0.5],

        "wd": [5e-4],

    },

    ("pubmed", "gat"):  {
        "layer":   [2, ],
        "hidden":  [64],
        "lr":      [0.005],
        "dropout": [0.2, ],
        "wd":      [5e-5],
    },

    ("pubmed", "gin"): {

        "layer": [2],

        "hidden": [64],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-4],

    },

    ("pubmed", "graphsage"): {

        "layer": [2, ],

        "hidden": [64],

        "lr": [0.001],

        "dropout": [0.2],

        "wd": [5e-4],

    },



    # ======================

    # 4) ACTOR

    # ======================

    ("actor", "gcn"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5,],  # dropout 영향 명확히 긍정적

        "wd": [5e-4],

    },

    ("actor", "gat"): {
        "layer":   [2, ],
        "hidden":  [64],
        "lr":      [0.005],
        "dropout": [0.2],
        "wd":      [0],
    },

    ("actor", "gin"): {

        "layer": [2],

        "hidden": [64],

        "lr": [0.01],

        "dropout": [0.2],

        "wd": [5e-4],

    },

    ("actor", "graphsage"): {

        "layer": [3],   # actor에서 GraphSAGE는 3 layer 우세

        "hidden": [128],

        "lr": [0.005],

        "dropout": [0.5],

        "wd": [5e-5],

    },



    # ======================

    # 5) OGBN-PRODUCTS

    # ======================

    ("ogbn-products", "gcn"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.01],

        "dropout": [0.5],

        "wd": [5e-5],

    },

    ("ogbn-products", "gat"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.005],

        "dropout": [0.2],

        "wd": [5e-5],

    },

    ("ogbn-products", "gin"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.5],

        "wd": [5e-5],

    },

    ("ogbn-products", "graphsage"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.005],

        "dropout": [0.2],

        "wd": [5e-5],

    },



    # ======================

    # 6) OGBN-ARXIV

    # ======================

    ("ogbn-arxiv", "gcn"): {

        "layer": [3],

        "hidden": [256, ],

        "lr": [0.01],

        "dropout": [0.3,],

        "wd": [5e-5, ],

    },

    ("ogbn-arxiv", "gat"): {
        "layer":   [2, ],
        "hidden":  [128, ],
        "lr":      [0.005,],
        "dropout": [0.2],
        "wd":      [0,],
    },

    ("ogbn-arxiv", "gin"): {

        "layer": [2],

        "hidden": [256],

        "lr": [0.001],

        "dropout": [0.2],
        "wd": [5e-5, ],

    },

    ("ogbn-arxiv", "graphsage"): {

        "layer": [3],

        "hidden": [256],

        "lr": [0.001],

        "dropout": [0.2],

        "wd": [5e-5],

    },
    ("texas", "gcn"): {
        "layer": [2, ],
        "hidden": [32,],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0, ]
    },

    ("texas", "graphsage"): {
        "layer": [2, ],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },

    ("cornell", "gcn"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },

    ("cornell", "graphsage"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("citeseer", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cora", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("pubmed", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
    ("cornell", "sgc"): {
        "layer": [2],
        "hidden": [32],
        "lr": [0.001],
        "dropout": [0.3],
        "wd": [0]
    },
}