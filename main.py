import argparse
import logging
import importlib
import types

from IPython import embed
import train

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s- %(filename)s:%(lineno)d - %(message)s')

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", "--cn", required=True)
    parser.add_argument("--cube-type", "--ct", required=False)
    parser.add_argument("--pretrain-model", "--pm", required=False)
    return parser

def reload_workspace(ws_ins):
    importlib.reload(train)
    new_class = train.Workspace
    for attr_name, attr_value in new_class.__dict__.items():
        if callable(attr_value):
            setattr(ws_ins, attr_name, types.MethodType(attr_value, ws_ins))


def main():
    # using py file as config
    import config # reload-able

    parser = get_args_parser()
    args = parser.parse_args()
    logging.info(f"args: {args}")

    config_name = args.config_name
    if not hasattr(config, config_name):
        logging.error(f"No config name in config.py: {config_name}")
        exit(-1)
    cfg = getattr(config, config_name)
    logging.info(f"cfg: {cfg}")

    workspace  = train.Workspace(cfg, args)
    embed() 
    # workspace.train()
    # workspace.infer()

if __name__ == "__main__":
    main()