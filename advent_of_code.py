
# %%
import math
import functools
import traceback
from typing import Dict, List, Tuple

def input_opener(filename, sep, as_type):
    try:
        with open(filename) as file:
            return [as_type(entry) for entry in file.read().split(sep)[:-1]]
    except Exception as e:
        print(traceback.format_exc())

# %%
# Day 1

def day_1(day_1_list: List) -> int:
    return functools.reduce(
        lambda accum, current_depth: (
            (current_depth, accum[1]) if
            accum[0]>=current_depth else
            (current_depth, accum[1]+1)
        ),
        day_1_list[1:],
        (day_1_list[0], 0)
    )[1]


i1 = input_opener("1.txt", "\n", int)
day_1(i1)

# %%
def day_1_2(day_1_2list: List) -> int:
    organise_inputs = {}
    for index, entry in enumerate(day_1_2list):
        window_size = 3
        sliding_window = list(val for val in range(index-1, index-1+window_size) if val > 0)
        for key in sliding_window:
            organise_inputs[key] = organise_inputs.get(key, 0) + entry
    return day_1(list(organise_inputs.values()))


day_1_2(i1)

# %%
def depth_calculator(accum: Dict, movement: Tuple) -> Dict:
    signal = -1 if movement[0] == 'up' else 1
    direction = "vertical" if movement[0] in ["up", "down"] else "horizontal"
    accum[direction] = accum[direction] + signal * movement[1]
    return accum

def day_2(day_2_list: List) -> int:
    mult_factors = functools.reduce(
        depth_calculator,
        day_2_list,
        {"vertical": 0, "horizontal":0}
    )
    return mult_factors["vertical"] * mult_factors["horizontal"]


i2 = [(entry.split(" ")[0], int(entry.split(" ")[1])) for entry in input_opener("2.txt", "\n", str)]
day_2(i2)

# %%
