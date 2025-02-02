import re
import json

from IPython import embed

# cstimer output file -> {idx -> scramble status}
#   scramble status: (scramble, scarmble status, prefer, ...)
def load_data_from_file(path, session_id, scramble_type, need_preprocess=False) -> list[int, tuple]:
    with open(path, 'r') as f:
        raw_data = json.load(f)
    ss_data = raw_data.get(f"session{session_id}", {})
    if not ss_data:
        print("[warning] not session data ")
        return
    ret_data = []
    for did, dt in enumerate(ss_data):
        pf = False
        if len(dt) > 4 and type(dt[4][-1]) == bool:
            pf = dt[4][-1]
        else:
            print(f'[warning] not prefer setting in session: {session_id}-{did}')

        status = SCRAMBLE_TYPE_TO_STATE_FUNC[scramble_type](dt[1])
        if need_preprocess:
            status = SCRAMBLE_TYPE_TO_PREPROCESS_FUNC[scramble_type](status)
        # fixed format:
        # [0]: scramble
        # [1]: scramble state
        # [2]: prefer against to last one
        ret_data.append((
            dt[1],
            status,
            pf
        ))
    # print(len(ss_data))
    # print(ret_data)
    return ret_data

'''
print to this form 
0 1 2        9
3 4 5 y2 10 11 12
6 7 8        13
'''
def pretty_print_clock(clock_status: list[int]):
    s = clock_status
    print(f"{s[0]} {s[1]} {s[2]}"+" "*8+f"{s[9]}")
    print(f"{s[3]} {s[4]} {s[5]}"+" y2 "+f"{s[10]} {s[11]} {s[12]}")
    print(f"{s[6]} {s[7]} {s[8]}"+" "*8+f"{s[13]}")

def clock_status_preprocess(clock_status: list[int]):
    return [s/11. for s in clock_status]

# due to the Rubiks's clock ops satisfy Abelian group
# let's write it out by hand
def clock_scramble_to_status(scramble_str: str) -> list[int]:
    move_array = [
        [ 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # UR
        [ 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], # DR
        [ 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # DL
        [ 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # UL
        [ 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # U
        [ 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], # R
        [ 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # D
        [ 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # L
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # ALL
        [11, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0], # UR
        [ 0, 0, 0, 0, 0, 0,11, 0, 0, 0, 0, 1, 1, 1], # DR
        [ 0, 0, 0, 0, 0, 0, 0, 0,11, 0, 1, 1, 0, 1], # DL
        [ 0, 0,11, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], # UL
        [11, 0,11, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], # U
        [11, 0, 0, 0, 0, 0,11, 0, 0, 1, 0, 1, 1, 1], # R
        [ 0, 0, 0, 0, 0, 0,11, 0,11, 0, 1, 1, 1, 1], # D
        [ 0, 0,11, 0, 0, 0, 0, 0,11, 1, 1, 1, 0, 1], # L
        [11, 0,11, 0, 0, 0,11, 0,11, 1, 1, 1, 1, 1]  # ALL
    ]
    move2idx = {
        "UR": 0,
        "DR": 1,
        "DL": 2,
        "UL": 3,
        "U": 4,
        "R": 5,
        "D": 6,
        "L": 7,
        "ALL": 8,
    }

    ret_status = [0] * 14
    
    y2_times = scramble_str.count("y2")
    if y2_times > 1:
        print(f"wrong clock scramble with y2 x {y2_times}")
        return []
    if y2_times == 1:
        front_scr, back_scr = scramble_str.split("y2")
        front_scr = front_scr.strip().split(" ")
        back_scr = back_scr.strip().split(" ")
    else:
        front_scr = scramble_str.strip().split(" ")
        back_scr = ""
    
    def exec_move(status, op, is_back=False):
        mt = re.match(r'^([A-Z]{1,3})(.*)', op)
        if not mt:
            print(f'error in exec_move with op:{op}')
            return []
        move = mt.group(1)
        step = mt.group(2)
        clk_wise = 1 if step[-1] == '+' else -1
        step = int(step[:-1])
        offset_idx = 9 if is_back else 0
        mv_vec = move_array[move2idx[move]+offset_idx]
        for i, mv_bias in enumerate(mv_vec):
            cur_s = status[i]
            status[i] = cur_s + clk_wise * step * mv_bias
    
    for op in front_scr:
        exec_move(ret_status, op)
        # print(op)
        # pretty_print_clock(ret_status)
    for op in back_scr:
        exec_move(ret_status, op, True)
        # print(op)
        # pretty_print_clock(ret_status)
    ret_status = [c%12 for c in ret_status]
    return ret_status

SCRAMBLE_TYPE_TO_STATE_FUNC = {
    "clock": clock_scramble_to_status,
}

SCRAMBLE_TYPE_TO_PREPROCESS_FUNC = {
    "clock": clock_status_preprocess,
}

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    load_data_from_file("./data/ye_clock_1.json", 1, "clock")
    
    status = clock_scramble_to_status("UR3+ DR5+ DL2- UL4- U4- R2+ D4- L4+ ALL2+ y2 U2- R1- D5- L4+ ALL3+")
    print(status)
    status = clock_scramble_to_status("UR3+ DR5+ DL2- UL4- U4- R2+ D4- L4+ ALL2+ ")
    print(status)
    status = clock_scramble_to_status("y2 UR3+ DR5+ DL2- UL4- U4- R2+ D4- L4+ ALL2+ ")
    print(status)
