import re
import json

from IPython import embed

def get_status_from_scrambles(scramble, scramble_type, need_preprocess=False) -> list:
    status = SCRAMBLE_TYPE_TO_STATE_FUNC[scramble_type](scramble)
    if need_preprocess:
        status = SCRAMBLE_TYPE_TO_PREPROCESS_FUNC[scramble_type](status)
    return status

# cstimer output file -> {idx -> scramble status}
#   scramble status: (scramble, scarmble status, prefer, ...)
def load_data_from_file(path, session_id, scramble_type, need_preprocess=False) -> list[tuple]:
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

        status = get_status_from_scrambles(dt[1], scramble_type, need_preprocess)
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

def nnn_status_preprocess(nnn_status: list[int]):
    return [s/5. for s in nnn_status]

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
        back_scr = []
    front_scr = [s for s in front_scr if s!=""]
    back_scr = [s for s in back_scr if s!=""]
    
    def exec_move(status, op, is_back=False):
        mt = re.match(r'^([A-Z]{1,3})(.*)', op)
        if not mt:
            print(f'error in exec_move with op:`{op}`')
            return
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

# nnn cube. copied from cstimer, make it into three parts:
# 1. cubeutil.parseScramble -> parse_nnn_scramble
# 2. image.nnnImage.doslice -> do_nnn_slice
# 3. image.nnnImage.getPosit -> nnn_scramble_to_status

nnn_scramble_reg = re.compile(r'^([\d]+(?:-\d+)?)?([FRUBLDfrubldzxySME])(?:([w])|&sup([\d]);)?([2\'])?$')

def parse_nnn_scramble(move_map: str, scramble: str="") -> list[list[int]]:
    moveseq = []
    moves = scramble.strip().split(' ')
    
    for move in moves:
        m = nnn_scramble_reg.match(move)
        if m is None:
            continue
        
        f = "FRUBLDfrubldzxySME".find(m.group(2))
        
        # 如果是中层转动
        if f > 14:
            p = "2'".find(m.group(5) or 'X') + 2
            f = [0, 4, 5][f % 3]
            moveseq.append([move_map.index("FRUBLD"[f]), 2, p])
            moveseq.append([move_map.index("FRUBLD"[f]), 1, 4 - p])
            continue
        
        w = (m.group(1) or '').split('-')
        w2 = int(w[1]) if len(w) > 1 else -1
        w = f < 12 and (int(w[0]) if w[0] else int(m.group(4) or 0) or ((m.group(3) == "w" or f > 5) and 2) or 1) or -1
        p = (f < 12 and 1 or -1) * ("2'".find(m.group(5) or 'X') + 2)
        # [face, width, clockwise step, idk]
        # 最后一个参数需要每步是形如 1-3R 这样的转动
        moveseq.append([move_map.index("FRUBLD"[f % 6]), w, p, w2])
    
    return moveseq


# f: face, [ D L B U R F ]
# d: which slice, in [0, size-1)
# q: [  2 ']
def do_nnn_slice(size: int, posit: list[int], f, d, q):
    s2 = size * size
    if f > 5:
        f -= 6
    for k in range(q):
        for i in range(size):
            if f == 0:
                f1 = 6 * s2 - size * d - size + i
                f2 = 2 * s2 - size * d - 1 - i
                f3 = 3 * s2 - size * d - 1 - i
                f4 = 5 * s2 - size * d - size + i
            elif f == 1:
                f1 = 3 * s2 + d + size * i
                f2 = 3 * s2 + d - size * (i + 1)
                f3 = s2 + d - size * (i + 1)
                f4 = 5 * s2 + d + size * i
            elif f == 2:
                f1 = 3 * s2 + d * size + i
                f2 = 4 * s2 + size - 1 - d + size * i
                f3 = d * size + size - 1 - i
                f4 = 2 * s2 - 1 - d - size * i
            elif f == 3:
                f1 = 4 * s2 + d * size + size - 1 - i
                f2 = 2 * s2 + d * size + i
                f3 = s2 + d * size + i
                f4 = 5 * s2 + d * size + size - 1 - i
            elif f == 4:
                f1 = 6 * s2 - 1 - d - size * i
                f2 = size - 1 - d + size * i
                f3 = 2 * s2 + size - 1 - d + size * i
                f4 = 4 * s2 - 1 - d - size * i
            elif f == 5:
                f1 = 4 * s2 - size - d * size + i
                f2 = 2 * s2 - size + d - size * i
                f3 = s2 - 1 - d * size - i
                f4 = 4 * s2 + d + size * i

            c = posit[f1]
            posit[f1] = posit[f2]
            posit[f2] = posit[f3]
            posit[f3] = posit[f4]
            posit[f4] = c

        if d == 0:
            for i in range(size // 2):
                for j in range((size - 1) // 2):
                    f1 = f * s2 + i + j * size
                    f3 = f * s2 + (size - 1 - i) + (size - 1 - j) * size
                    if f < 3:
                        f2 = f * s2 + (size - 1 - j) + i * size
                        f4 = f * s2 + j + (size - 1 - i) * size
                    else:
                        f4 = f * s2 + (size - 1 - j) + i * size
                        f2 = f * s2 + j + (size - 1 - i) * size

                    c = posit[f1]
                    posit[f1] = posit[f2]
                    posit[f2] = posit[f3]
                    posit[f3] = posit[f4]
                    posit[f4] = c


def nnn_scramble_to_status(size: int, scramble: str) -> list[int]:
    cnt = 0
    posit = []
    for i in range(6):
        for f in range(size * size):
            posit.append(i)
            cnt += 1

    # moves = cubeutil.parse_scramble(moveseq, "DLBURF", True)
    moves = parse_nnn_scramble("DLBURF", scramble)
    for s in range(len(moves)):
        for d in range(moves[s][1]):
            # do_nnn_slice(moves[s][0], d, moves[s][2], size)
            do_nnn_slice(size, posit, moves[s][0], d, moves[s][2])
        if moves[s][1] == -1:
            for d in range(size - 1):
                do_nnn_slice(size, posit, moves[s][0], d, -moves[s][2])
            do_nnn_slice(size, posit, (moves[s][0] + 3) % 6, 0, moves[s][2] + 4)
    
    return posit


SCRAMBLE_TYPE_TO_STATE_FUNC = {
    "clock": clock_scramble_to_status,
    "222": lambda s: nnn_scramble_to_status(2, s),
    "333": lambda s: nnn_scramble_to_status(3, s),
    "444": lambda s: nnn_scramble_to_status(4, s),
    "555": lambda s: nnn_scramble_to_status(5, s),
    "666": lambda s: nnn_scramble_to_status(6, s),
    "777": lambda s: nnn_scramble_to_status(7, s),
}

SCRAMBLE_TYPE_TO_PREPROCESS_FUNC = {
    "clock": clock_status_preprocess,
    "222": nnn_status_preprocess,
    "333": nnn_status_preprocess,
    "444": nnn_status_preprocess,
    "555": nnn_status_preprocess,
    "666": nnn_status_preprocess,
    "777": nnn_status_preprocess,
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
    status = clock_scramble_to_status("")
    print(status)

    print("------------nnn-----------")
    move_seq = parse_nnn_scramble("URFDLB", "U' F L R B D u l 1-3Fw")
    print(move_seq)
    
    # URFDLB
    status = nnn_scramble_to_status(3, "y3")
    print(status)
    status = nnn_scramble_to_status(3, "R")
    print(status)
    status = nnn_scramble_to_status(3, "U' F D' R U L' D R F2 U2 R2 D' B2 R2 D F2 L2 B2 L2 D'")
    print(status)

    status = SCRAMBLE_TYPE_TO_STATE_FUNC["333"]("U' F D' R U L' D R F2 U2 R2 D' B2 R2 D F2 L2 B2 L2 D'")
    print(status)
