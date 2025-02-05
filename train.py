import os
import onnx.checker
from tqdm import tqdm
import logging
import itertools

import onnx
import onnxruntime
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from models import Policy
from util import SCRAMBLE_TYPE_TO_STATE_FUNC
from dataset import get_dataset
import loss

def get_config(cfg, args, key):
    if hasattr(args, key) and (attr := getattr(args, key)):
        return attr
    elif key in cfg:
        return cfg[key] 
    logging.warning(f'There is not setting for `{key}`')
    return None



class Workspace:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        
        # 1. use dataset to init dataloader
        self.cube_type = self.get_config("cube_type")
        dataset = get_dataset(self)
        if not dataset:
            logging.error(f"There is no dataset for {self.get_config('dataset')}")
        else:
            self.dataloader = DataLoader(
                dataset, shuffle=True, 
                num_workers=self.get_config("dataloader_workers") , 
                batch_size=self.get_config("batch_size")
                )
            # make it infinite
            self.dataloader = itertools.cycle(self.dataloader)

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
                chosen_states_ts, reject_states_ts = next(self.dataloader)
                chosen_states_ts = chosen_states_ts.to(device)
                reject_states_ts = reject_states_ts.to(device)
                # infer twice
                chosen_rewards = self.network(chosen_states_ts)
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

    # trans to onnx model
    def to_onnx(self):
        x = SCRAMBLE_TYPE_TO_STATE_FUNC[self.cube_type]("")
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.float32)
        # embed() # debug
        x = x.unsqueeze_(0) # (batch_idx, state_dim)
        y = self.network(x) # (batch_idx, 1)
        logging.info(f"to onnx, input dim:{x.shape}, output dim: {y.shape}")
        onnx_path = f"{self.exp_dir}/exp_onnx_{len(self.losses)}.onnx"
        torch.onnx.export(
            self.network,
            x,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "state_num"},
            }
        )
        # onnx load test
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}

        import time
        beg_time = time.time()
        for _ in range(10000):
            ort_outs = ort_session.run(None, ort_inputs)
        logging.info(f"onnx test infer time: {time.time() - beg_time}")
        np.testing.assert_allclose(y.detach().numpy(), ort_outs[0], rtol=1e-3, atol=1e-5)
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


