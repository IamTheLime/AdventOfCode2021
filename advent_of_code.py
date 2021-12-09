

import math
import functools
import os
from pprint import pprint
import re
import traceback
from typing import Dict, List, Tuple

os.chdir('/Users/tiagolima/Documents/personal_repos/AdventOfCode2021')
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
        if entry == sep:
            if current_sublist != []:
                list_of_sublists.append(current_sublist)
            current_sublist = []
        else:
            current_sublist.append(entry)
    if len(current_sublist) != 0:
        list_of_sublists.append(current_sublist)

    return list_of_sublists

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
    raw_boards = list_of_lists_splitter(inpt[1:], "")
    board_sizes = {"row_len": len(raw_boards[0]), "col_len": len(re.split("\s+", raw_boards[0][0]))}
    boards = {}
    for index, board in enumerate(raw_boards):
        boards[f"board_{index}"] = {"board_number": index, "board": {}}
        for row_i, row  in enumerate(board):
            for col_i, column in enumerate(re.split("\s+", row.lstrip(" "))):
                boards[f"board_{index}"]["board"][column] = (row_i, col_i)

    return bingo_numbers, boards, board_sizes

import copy
def day_4(inpt, only_return_last = False):
    bingo_numbers, boards, board_sizes = split_into_boards(inpt)
    bingos =  []
    board_helper = {}
    global_results = []
    for entry in bingo_numbers:
        for board_name, board in boards.items():
            row_col = board["board"].get(entry, None)
            if row_col:
                row, column = row_col
                board_helper[f"{board_name}_tagged_entries"] = set.union(board_helper.get(f"{board_name}_tagged_entries", set()), {int(entry)})
                board_helper[f"{board_name}_row_count_{row}"] = board_helper.get(f"{board_name}_row_count_{row}", 0) + 1
                board_helper[f"{board_name}_row_{row}"] = board_helper.get(f"{board_name}_row_{row}", []) + [int(entry)]
                board_helper[f"{board_name}_col_count_{column}"] = board_helper.get(f"{board_name}_col_count_{column}", 0) + 1
                board_helper[f"{board_name}_col_{column}"] = board_helper.get(f"{board_name}_col_{column}", []) + [int(entry)]

            if board_helper.get(f"{board_name}_row_count_{row}") == board_sizes["row_len"]:
                bingos.append((board_name,"row", row, entry, board_helper[f"{board_name}_tagged_entries"]))
            if board_helper.get(f"{board_name}_col_count_{column}") == board_sizes["col_len"]:
                bingos.append((board_name,"col", column, entry, board_helper[f"{board_name}_tagged_entries"]))
    if bingos != []:
        if not only_return_last:
            bingo = bingos[0]
        else:
            bingoset = set()
            latest_bingo = None
            for bingo in bingos:
                if bingo[0] not in bingoset:
                    bingoset.add(bingo[0])
                    latest_bingo=bingo
                if len(bingoset) == len(boards):
                    break
            bingo=latest_bingo

        return sum([
            int(key) for key in boards[bingo[0]]["board"]
            if int(key) not in bingo[4]
        ]) * int(bingo[3])

i4 = input_opener("4.txt", "\n", str)
print(day_4(i4))
print(day_4(i4,True))
