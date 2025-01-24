from models import Policy
import argparse

class Workspace:
    def __init__(self, cfg, args):

        pass
    # TODO: implement training loop

if __name__ == "__main__":    
    #  using py file as config
    import config as cfg
    parser = argparse.ArgumentParser()

    workspace  = Workspace(cfg, parser.parse_args())
