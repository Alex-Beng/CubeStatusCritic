from models import Policy
import argparse

def get_config(cfg, args, key):
    if hasattr(args, key):
        return getattr(args, key)
    elif key in cfg:
        return cfg[key] 

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", "--cn", required=True)
    return parser

class Workspace:
    def __init__(self, cfg, args):
        # TODO: load data -> process to list of (scramble etc..)
        pass
    
    # TODO: trainging loop
    def train(self):
        pass
    
    # TODO: infer scrambles
    def infer(self):
        pass

    # TODO: trans to onnx model
    def to_onnx(self):
        pass

if __name__ == "__main__":    
    #  using py file as config
    import config

    parser = get_args_parser()
    args = parser.parse_args()
    print(args)

    config_name = args.config_name
    if not hasattr(config, config_name):
        print(f"[error] not config name in config.py: {config_name}")
        exit(-1)
    cfg = getattr(config, config_name)
    print(cfg)

    workspace  = Workspace(cfg, args)
