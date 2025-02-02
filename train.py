import os
import argparse
from tqdm import tqdm
import logging
import random

from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

from models import Policy
from util import load_data_from_file, SCRAMBLE_TYPE_TO_STATE_FUNC
import loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s- %(filename)s:%(lineno)d - %(message)s')

def get_config(cfg, args, key):
    if hasattr(args, key) and (attr := getattr(args, key)):
        return attr
    elif key in cfg:
        return cfg[key] 
    logging.warning(f'There is not setting for `{key}`')
    return None

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", "--cn", required=True)
    parser.add_argument("--cube-type", "--ct", required=False)
    parser.add_argument("--pretrain-model", "--pm", required=False)
    return parser

class Workspace:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        
        # 1. load data -> process to list of (scramble etc..)
        self.check_cube_type = self.get_config("check_cube_type")
        if self.get_config("cube_type"):
            self.cube_type = self.get_config("cube_type")

        if self.get_config("data_files"):
            self.load_data()
            logging.info(f"loaded data sample: {self.scrambles[:2]}")
            logging.info(f"loaded data skip idx: {self.head_scr_idx}")

        # 1.5 output dir
        self.exp_dir = self.get_config("exp_dir")
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            logging.info(f"create {self.exp_dir}")

        # 2. init hyper params & network
        global device
        device = self.get_config("device")
        
        self.epoch = self.get_config("epoch")
        self.batch_size = self.get_config("batch_size")
        self.lr = self.get_config("lr")
        
        self.network = Policy(
            self.get_config("input_size"),
            1,
            self.get_config("hidden_layer_size"),
            self.get_config("hidden_depth"),
        )
        self.network.to(device)
        self.optimizer = optim.Adam(list(self.network.parameters()), lr=self.lr)
        loss_fn = self.get_config("loss")
        self.loss_margin = self.get_config("loss_margin")
        if not hasattr(loss, loss_fn):
            logging.error(f"There is not impl for loss function: {loss_fn}")
            exit(-2)
        self.loss = getattr(loss, loss_fn)()

        if self.get_config("pretrain_model"):
            logging.info("trying to load pretrain model")
            self.load_model()


        # book keeping
        self.plot_window = self.get_config("plot_window")
        self.train_steps = [] # for plot
        self.losses = []
    
    # region-----------helper function---------------
    
    def init_and_get_loss_margin(self):
        if not hasattr(self, "loss_margin_ts"):
            margin_ts = torch.tensor(self.loss_margin, dtype=torch.float32).to(device)
            self.loss_margin_ts = margin_ts
        return self.loss_margin_ts

    def init_and_get_sample_idx(self):
        if not hasattr(self, "sample_idx"):
            sample_idx = list(range(len(self.scrambles)-1)) # tail is out of scope
            for idx in self.head_scr_idx:
                sample_idx.pop(idx)
            self.sample_idx = sample_idx
        return self.sample_idx

    # store the data in the memo
    def sample_pairs(self, pair_num):
        idxs = self.init_and_get_sample_idx()
        # TODO: support weights
        chosen_pairs = random.choices(idxs, k=pair_num)
        
        # chosen idx & chosen idx+1 -> chosen & reject
        chosen_states = []
        reject_states = []
        for idx in chosen_pairs:
            c_s = self.scrambles[idx]
            r_s = self.scrambles[idx+1]
            if r_s[2] == True:
                c_s, r_s = r_s, c_s
            chosen_states.append(c_s[1])
            reject_states.append(r_s[1])
        return chosen_states, reject_states
        
    def load_data(self):
        data_dir = self.get_config("data_dir")
        data_files = self.get_config("data_files")
        
        self.scrambles = []
        self.head_scr_idx = set() # exclude the first scramble
        for file in data_files:
            file_name = file["name"]
            file_ssid = file["session_id"]
            file_type = file["type"]

            if not hasattr(self, "cube_type"):
                logging.info(f"using first data_file's cube type: {file_type}")
                self.cube_type = file_type
            if self.check_cube_type and file_type != self.cube_type:
                logging.warning(f"file: {file_name}, ssid: {file_ssid} has wrong cube type. Setting `check_cube_type` = False to ignore this check")
                continue
            data_path = f"{data_dir}/{file_name}"
            
            data = load_data_from_file(data_path, file_ssid, self.cube_type)
            logging.info(f"read {len(data)} from {data_path} ssid:{file_ssid}")
            
            self.head_scr_idx.add(len(self.scrambles))
            self.scrambles += data

    def get_config(self, key):
        return get_config(self.cfg, self.args, key)

    # endregion------------------

    # region-------------interface--------------
    def train(self):
        logging.info("begin to train...")
        self.network.train()
        with tqdm(range(self.epoch)) as tepoch:
            for i in tepoch:
                self.optimizer.zero_grad()
                chosen_states, reject_states = self.sample_pairs(self.batch_size)
                # infer twice
                chosen_states = np.array(chosen_states)
                chosen_states_ts = torch.tensor(chosen_states, dtype=torch.float32).to(device)
                chosen_rewards = self.network(chosen_states_ts)
                reject_states_ts = torch.tensor(reject_states, dtype=torch.float32).to(device)
                reject_rewards = self.network(reject_states_ts)
                
                # clac loss
                epoch_loss = self.loss(
                    chosen_rewards, 
                    reject_rewards,
                    self.init_and_get_loss_margin()
                )
                
                # embed() # import os; os._exit(0)
                # step
                epoch_loss.backward()
                self.optimizer.step()
                
                # plot
                self.train_steps.append(i)
                self.losses.append(epoch_loss.item())
                if i % self.plot_window == 0:
                    plt.clf(); plt.cla()
                    plt.yscale('log')
                    plt.plot(self.train_steps, self.losses, label="loss", color="blue")
                    plt.legend()
                    plt.savefig(f"{self.exp_dir}/train_loss.png")
                tepoch.set_postfix(loss=epoch_loss.item() / (tepoch.n + 1))
        self.save_model()
    
    def infer(self):
        if len(self.losses) == 0:
            logging.warning(f"infer model may not being trained")
        self.network.eval()
        print("Enter infer loop, type `quit` or `exit` to quit the loop")
        while True:
            scramble = input("input scramble >").strip()
            if scramble in {"quit", "exit"}:
                print("quit infer loop")
                break
            try:
                status = SCRAMBLE_TYPE_TO_STATE_FUNC[self.cube_type](scramble)
                status = np.array(status)
                status = torch.tensor(status, dtype=torch.float32).to(device)
                reward = self.network(status)
                print(reward)
            except Exception as e:
                print(e)
            finally:
                if locals().get("e", None):
                    print(e)
        pass

    # TODO: trans to onnx model
    def to_onnx(self):
        
        pass

    def save_model(self):
        torch.save(self.network.state_dict(), f"{self.exp_dir}/model_{len(self.losses)}.bin")

    def load_model(self):
        if not self.get_config("pretrain_model"):
            logging.warning("load model with not pretrain_model setting")
            return
        # try to load {data_dir}/model_path; model_path; {exp_dir}/model_path
        def _load_model(path):
            self.network.load_state_dict(torch.load(path))
        model_path = self.get_config("pretrain_model")
        data_dir = self.get_config("data_dir")
        exp_dir = self.get_config("exp_dir")
        if os.path.isfile(model_path):
            logging.info(f"load model from {model_path}")
            _load_model(model_path)
        elif os.path.isfile(f"{data_dir}/{model_path}"):
            logging.info(f"load model from {data_dir}/{model_path}")
            _load_model(f"{data_dir}/{model_path}")
        elif os.path.isfile(f"{exp_dir}/{model_path}"):
            logging.info(f"load model from {exp_dir}/{model_path}")
            _load_model(f"{exp_dir}/{model_path}")
        else:
            logging.warning(f"invalid model_path: {model_path}, check")

    # endregion------------------
if __name__ == "__main__":    
    #  using py file as config
    import config

    parser = get_args_parser()
    args = parser.parse_args()
    logging.info(f"args: {args}")

    config_name = args.config_name
    if not hasattr(config, config_name):
        logging.error(f"No config name in config.py: {config_name}")
        exit(-1)
    cfg = getattr(config, config_name)
    logging.info(f"cfg: {cfg}")

    workspace  = Workspace(cfg, args)
    embed() 
    # workspace.train()
    # workspace.infer()
