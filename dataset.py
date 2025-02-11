import logging
from copy import deepcopy
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from util import load_data_from_file, get_status_from_scrambles

class BasicDataset(Dataset):
    def __init__(self, workspace: 'Workspace'):
        self.workspace = workspace

        # 1. load data -> process to list of (scramble etc..)
        self.check_cube_type = workspace.get_config("check_cube_type")
        if workspace.get_config("cube_type"):
            self.cube_type = workspace.get_config("cube_type")

        if workspace.get_config("data_files"):
            self.load_data()
            logging.info(f"loaded data sample: {self.scrambles[:2]}")
            logging.info(f"loaded data skip idx: {self.head_scr_idx}")

    def load_data(self):
        data_dir = self.workspace.get_config("data_dir")
        data_files = self.workspace.get_config("data_files")

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

    def init_and_get_sample_idx(self):
        if not hasattr(self, "sample_idx"):
            sample_idx = list(range(len(self.scrambles)-1)) # tail is out of scope
            for idx in self.head_scr_idx:
                sample_idx.pop(idx)
            self.sample_idx = sample_idx
        return self.sample_idx

    def __len__(self):
        raise Exception("No implement __len__")

    def __getitem__(self, index):
        raise Exception("No implement __getitem__")

class ClockDataset(BasicDataset):
    def __init__(self, workspace: 'Workspace'):
        super().__init__(workspace)

        self.flip_prob = workspace.get_config("aug_params").get("flip_prob", 0.0) # 0.0
        if self.flip_prob > 0.0:
            # flip state list
            self.flip_data()
    
    def __len__(self):
        return len(self.scrambles) * 2
    
    def __getitem__(self, __idx):
        idxs = self.init_and_get_sample_idx()
        # TODO: support weights
        chosen_pair = random.choice(idxs)

        # chosen idx & chosen idx+1 -> chosen & reject
        idx = chosen_pair
        if random.random() > self.flip_prob:
            c_s = self.scrambles[idx]
            r_s = self.scrambles[idx+1]
        else:
            c_s = self.flip_scrambles[idx]
            r_s = self.flip_scrambles[idx+1]
        
        if r_s[2] == True:
            c_s, r_s = r_s, c_s 

        # to tensor
        chosen_state = np.array([c_s[1]])
        reject_state = np.array([r_s[1]])
        chosen_state_ts = torch.tensor(chosen_state, dtype=torch.float32)
        reject_state_ts = torch.tensor(reject_state, dtype=torch.float32)
        return chosen_state_ts, reject_state_ts
        
    def flip_data(self):
        def _flip_state(state):
            new_state = list(deepcopy(state))
            new_state[1] = [
                (12-state[1][0])%12, state[1][9], (12-state[1][2])%12,
                state[1][10], state[1][11], state[1][12], 
                (12-state[1][6])%12, state[1][13], (12-state[1][8])%12,
                state[1][1], 
                state[1][3], state[1][4], state[1][5], 
                state[1][7],
            ]
            return tuple(new_state)
        flipped_scrambles = []
        for scr in self.scrambles:
            flipped_scrambles.append(_flip_state(scr))
        self.flip_scrambles = flipped_scrambles

class NNNDataset(BasicDataset):
    def __init__(self, workspace: 'Workspace'):
        super().__init__(workspace)

        # due to * 24 may cost too much memory, don't do data aug here
        self.pre_scrs = workspace.get_config("aug_params").get("pre_scrambles", [""])

    def __len__(self):
        return len(self.scrambles) * 24

    def __getitem__(self, __idx):
        idxs = self.init_and_get_sample_idx()
        # TODO: support weights
        chosen_pair = random.choice(idxs)
        chosen_pre_scr = random.choice(self.pre_scrs)

        # chosen idx & chosen idx+1 -> chosen & reject
        idx = chosen_pair
        c_s = self.scrambles[idx]
        r_s = self.scrambles[idx+1]
        if r_s[2] == True:
            c_s, r_s = r_s, c_s
        
        n_c_s = [
            c_s[0],
            # TODO: 接入need_preprocess
            get_status_from_scrambles(chosen_pre_scr + c_s[0], self.workspace.cube_type),
            c_s[2]
        ]
        n_r_s = [
            r_s[0],
            # TODO: 接入need_preprocess
            get_status_from_scrambles(chosen_pre_scr + r_s[0], self.workspace.cube_type),
            r_s[2]
        ]

        # to tensor
        chosen_state = np.array([n_c_s[1]])
        reject_state = np.array([n_r_s[1]])
        chosen_state_ts = torch.tensor(chosen_state, dtype=torch.float32)
        reject_state_ts = torch.tensor(reject_state, dtype=torch.float32)
        return chosen_state_ts, reject_state_ts


SCRAMBLE_TYPE_TO_DATASET = {
    "clock": ClockDataset,
    "222": NNNDataset,
    "333": NNNDataset,
    "444": NNNDataset,
    "555": NNNDataset,
    "666": NNNDataset,
    "777": NNNDataset,
}

def get_dataset(workspace: 'Workspace'):
    cube_type = workspace.cube_type
    if cube_type not in SCRAMBLE_TYPE_TO_DATASET:
        logging.error(f"cube_type: {cube_type} not supported")
        return None
    
    return SCRAMBLE_TYPE_TO_DATASET[cube_type](workspace)
