clock_config = {
    # data relate
    "data_dir": "./data",
    "data_files": [ # set it empty for not loading data
        {
            "name": "ye_clock_1.json",
            "session_id": 1,
            "type": "clock"
        },
        {
            "name": "jiang_clock_1.json",
            "session_id": 1,
            "type": "clock"
        }
    ],
    "check_cube_type": False,
    "cube_type": "clock",
    
    # training relate
    "exp_dir": "./training/clock",
    "device": "cpu",

    "epoch": 5000,
    "batch_size": 128,
    "lr": 1e-4,
    "loss": "PairWiseLoss", # or use LogExpLoss
    # "loss": "LogExpLoss",
    "loss_margin": 10.0,

    "input_size": 14,
    "hidden_depth": 3,
    "hidden_layer_size": 64,
    
    "plot_window": 100,

    # "pretrain_model": "model_5000.bin" # set it to load pre-train model
}