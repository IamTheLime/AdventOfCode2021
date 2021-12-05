

import math
import functools
import traceback
from typing import Dict, List, Tuple

# Just some aux shit

def input_opener(filename, sep, as_type):
    try:
        with open(filename) as file:
            return [as_type(entry) for entry in file.read().split(sep)[:-1]]
    except Exception as e:
        print(traceback.format_exc())

def list_of_lists_splitter(lst, sep):
    list_of_sublists = []
    current_sublist = []
    for entry in lst:
        if entry == "sep":
            list_of_sublists.append(current_sublist)
            current_sublist = []
        else:
            current_sublist.append(entry)


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
print(day_1(i1))

# Day 2

def day_1_2(day_1_2list: List) -> int:
    organise_inputs = {}
    for index, entry in enumerate(day_1_2list):
        window_size = 3
        sliding_window = list(val for val in range(index-1, index-1+window_size) if val > 0)
        for key in sliding_window:
            organise_inputs[key] = organise_inputs.get(key, 0) + entry
    return day_1(list(organise_inputs.values()))


print(day_1_2(i1))

# Day 3

def depth_calculator(accum: Dict, movement: Tuple) -> Dict:
    signal = -1 if movement[0] == 'up' else 1
    direction = "vertical" if movement[0] in ["up", "down"] else "horizontal"
    accum[direction] = accum[direction] + signal * movement[1]
    return accum

def aim_calculator(accum: Dict, movement: Tuple) -> Dict:
    signal = -1 if movement[0] == 'up' else 1
    if movement[0] in ["up", "down"]:
        accum["aim"] = accum["aim"] + signal * movement[1]
    else:
        accum["horizontal"] = accum["horizontal"] + signal * movement[1]
        accum["vertical"] = accum["vertical"] +  accum["aim"] * movement[1]

    return accum


def day_2(day_2_list: List, calculator) -> int:
    mult_factors = functools.reduce(
        calculator,
        day_2_list,
        {"vertical": 0, "horizontal":0, "aim": 0}
    )
    return mult_factors["vertical"] * mult_factors["horizontal"]


i2 = [(entry.split(" ")[0], int(entry.split(" ")[1])) for entry in input_opener("2.txt", "\n", str)]
print(day_2(i2, depth_calculator))
print(day_2(i2, aim_calculator))

# Day 4

def get_mask(matrix) -> Tuple:
    index_count = {}
    for row in matrix:
        for row_index, bit in enumerate(row):
            current_icount = index_count.get(
                f"row_index_{row_index}",
                {"bit_0": 0, "bit_1": 0}
            )
            current_icount["bit_0"] += 1 if bit == 0 else 0
            current_icount["bit_1"] += 1 if bit == 1 else 0
            index_count[f"row_index_{row_index}"] = current_icount
    most_common_mask = []
    least_common_mask = []
    for value in index_count.values():
        most_common_mask.append("1" if value["bit_1"] >= value["bit_0"] else "0")
        least_common_mask.append("0" if value["bit_0"] <= value["bit_1"] else "1")
    return ''.join(most_common_mask), ''.join(least_common_mask)

def day_3(day_3_matrix: List) -> int:
    mc, lc = get_mask(day_3_matrix)
    return int(mc, 2) * int(lc, 2)

def day_3_1(day_3_matrix_original) -> int:
    min_m = day_3_matrix_original
    max_m = day_3_matrix_original

    for i in range(0, len(day_3_matrix_original[0])):
        mcmax, _ = get_mask(max_m)
        _, lcmin = get_mask(min_m)

        max_m = [ value for value in max_m if value[i] == int(mcmax[i]) or len(max_m) == 1]
        min_m = [ value for value in min_m if value[i] == int(lcmin[i]) or len(min_m) == 1]

        oxygen = int(''.join([str(val) for val in max_m[0]]), 2)
        co2 = int(''.join([str(val) for val in min_m[0]]), 2)

    return oxygen * co2


i3 = input_opener("3.txt", "\n", lambda x: [int(y) for y in list(x)])
print(day_3(i3))
print(day_3_1(i3))

# Day 4

def split_into_boards(inpt: List[str]) -> Tuple:
    bingo_numbers = inpt[0].split(",")
    boards = {}


    num_boards = 0

    boards = list_of_lists_splitter(inpt[1,:], ["\n"])
    for index, board in boards:
        print(index, board)
        # num_boards += 1 if line[0] == "\n" else 0
        # row_num += -1 if line[0] == "\n" else 1
        # for num in line.split(" "):
        #     board


    # return bingo_numbers


def day_4_1(inpt):
    split_into_boards(inpt)

i4 = input_opener("4.txt", "\n", str)
day_4_1(i4)
