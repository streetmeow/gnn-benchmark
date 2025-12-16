# run_settings/cpf_teacher_table.py

CPF_TEACHER_TABLE = {
    # ======================================================
    # CORA
    # ======================================================
    # gcn_cora_lr0.005_ly3_wd5e-05_dr0.2_hd128 / 82
    ("cora", "gcn"): {
        "model": "gcn",
        "checkpoint": "/app/output_logs/2025-11-29_21-53-47/best_checkpoint.pth",
        "lr": 0.005,
        "num_layers": 3,
        "weight_decay": 5e-5,
        "dropout": 0.2,
        "hidden_dim": 128,
    },

    # sgc_cora_lr0.01_ly1_wd0_dr0.0_hd32 / 78.3
    ("cora", "sgc"): {
        "model": "sgc",
        "checkpoint": "/app/output_logs/2025-12-13_21-28-32/best_checkpoint.pth",
        "lr": 0.01,
        "num_layers": 1,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "hidden_dim": 32,
    },

    # ======================================================
    # PUBMED
    # ======================================================
    # gcn_pubmed_lr0.01_ly2_wd0.0005_dr0.5_hd128 / 78.9
    ("pubmed", "gcn"): {
        "model": "gcn",
        "checkpoint": "/app/output_logs/2025-11-29_22-14-08/best_checkpoint.pth",
        "lr": 0.01,
        "num_layers": 2,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "hidden_dim": 128,
    },

    # sgc_pubmed_lr0.005_ly1_wd0_dr0.0_hd64 / 73.2
    ("pubmed", "sgc"): {
        "model": "sgc",
        "checkpoint": "/app/output_logs/2025-12-13_21-28-46/best_checkpoint.pth",
        "lr": 0.005,
        "num_layers": 1,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "hidden_dim": 64,
    },

    # ======================================================
    # CITESEER
    # ======================================================
    # gcn_citeseer_lr0.01_ly2_wd0.0005_dr0.5_hd128 / 69.8
    ("citeseer", "gcn"): {
        "model": "gcn",
        "checkpoint": "/app/output_logs/2025-11-25_19-15-42/best_checkpoint.pth",
        "lr": 0.01,
        "num_layers": 2,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "hidden_dim": 128,
    },

    # sgc_citeseer_lr0.01_ly1_wd0_dr0.0_hd32 / 67.4
    ("citeseer", "sgc"): {
        "model": "sgc",
        "checkpoint": "/app/output_logs/2025-12-13_21-28-40/best_checkpoint.pth",
        "lr": 0.01,
        "num_layers": 1,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "hidden_dim": 32,
    },

    # ======================================================
    # CORNELL
    # ======================================================
    # gcn_cornell_lr0.001_ly2_wd0.0005_dr0.5_hd32 / 48.6
    ("cornell", "gcn"): {
        "model": "gcn",
        "checkpoint": "/app/output_logs/2025-12-05_17-50-55/best_checkpoint.pth",
        "lr": 0.001,
        "num_layers": 2,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "hidden_dim": 32,
    },

    # sgc_cornell_lr0.01_ly1_wd0_dr0.0_hd16 / 40.54
    ("cornell", "sgc"): {
        "model": "sgc",
        "checkpoint": "/app/output_logs/2025-12-13_21-35-29/best_checkpoint.pth",
        "lr": 0.01,
        "num_layers": 1,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "hidden_dim": 16,
    },

    # ======================================================
    # TEXAS
    # ======================================================
    # gcn_texas_lr0.02_ly2_wd5e-06_dr0.7_hd128 / 67.56
    ("texas", "gcn"): {
        "model": "gcn",
        "checkpoint": "/app/output_logs/2025-12-05_15-51-20/best_checkpoint.pth",
        "lr": 0.02,
        "num_layers": 2,
        "weight_decay": 5e-6,
        "dropout": 0.7,
        "hidden_dim": 128,
    },

    # sgc_texas_lr0.01_ly1_wd0_dr0.0_hd16 / 64.86
    ("texas", "sgc"): {
        "model": "sgc",
        "checkpoint": "/app/output_logs/2025-12-13_21-28-18/best_checkpoint.pth",
        "lr": 0.01,
        "num_layers": 1,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "hidden_dim": 16,
    },
}
