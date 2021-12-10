

import math
import functools
from itertools import product
import os
from pprint import pprint
import re
import traceback
from typing import Dict, List, Tuple


def input_opener(filename, sep, as_type):
    try:
        with open(filename) as file:
            inpt = file.read()
            inpt = inpt[0:-1] if inpt[-1] == "\n" else inpt
            return [as_type(entry) for entry in inpt.split(sep)]
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


def get_ranges(inpt, count_diagonals):
    ranges = []
    for line in inpt:
        m = re.match(r"(?P<x1>[0-9]+),(?P<y1>[0-9]+) \-\> (?P<x2>[0-9]+),(?P<y2>[0-9]+)", line)
        entry_dict = m.groupdict()
        (x1, x2, y1, y2) = (int(entry_dict["x1"]), int(entry_dict["x2"]), int(entry_dict["y1"]), int(entry_dict["y2"]))
        if x1 == x2:
            ranges += list(product([x1], list(range(min(y1,y2), max(y1+1,y2+1)))))
        elif y1 == y2:
            ranges += list(product(list(range(min(x1,x2), max(x1+1,x2+1))), [y1]))
        elif count_diagonals and (m := (y2 - y1) / (x2 - x1)) in [1,-1]:
            start_x, start_y, end_x, end_y = (x1, y1, x2, y2) if x1 < x2 else (x2, y2, x1, y1)
            while (start_x, start_y) <= (end_x, end_y):
                ranges.append((start_x, start_y))
                start_x = start_x +1
                start_y = start_y + m * 1
    return(ranges)

def day_5(inpt, count_diagonals=False):
    overlay = {}
    for entry in get_ranges(inpt, count_diagonals):
        overlay[entry] = overlay.get(entry,0) + 1
    return functools.reduce(lambda acc, curr_value: acc + (1 if curr_value > 1 else 0) , overlay.values() ,0)

i5 = input_opener("5.txt", "\n", str)
print(day_5(i5))
print(day_5(i5,True))


def day_6(inpt, days):
    current_num_fishes = len(inpt)
    spawn_scheduler = {}
    for fish_counter in inpt:
        spawn_scheduler[fish_counter + 1] = spawn_scheduler.get(fish_counter + 1, 0) + 1

    for day in range(1, days+1):
        births = spawn_scheduler.get(day, 0)
        current_num_fishes += births
        spawn_scheduler[day + 8 + 1] =   spawn_scheduler.get(day + 8 + 1, 0) + births
        spawn_scheduler[day + 6 + 1] =   spawn_scheduler.get(day + 6 + 1, 0) + births

    return current_num_fishes



i6 = input_opener("6.txt", ",", int)
print(day_6(i6, 80))
print(day_6(i6, 256))

memo_cache = {}
def get_costs(pos_diff):
    if pos_diff in memo_cache:
        return memo_cache[pos_diff]

    retval = 0
    for i in range(pos_diff,0,-1):
        if i in memo_cache:
            memo_cache[pos_diff] = memo_cache[i] + retval
            return memo_cache[pos_diff]
        else:
            retval += i
    return retval


def day_7(inpt, part = "1"):
    low, high = min(inpt), max(inpt)

    min_sum = math.inf

    if part == "1":
        for position in range(low, high+1):
            curr_sum = sum(abs(position -  int(crab)) for crab in inpt)
            min_sum = curr_sum if curr_sum < min_sum else min_sum
    else:
        for position in range(low, high+1):
            curr_sum = sum(get_costs(abs(position - int(crab))) for crab in inpt)
            min_sum = curr_sum if curr_sum < min_sum else min_sum

    return min_sum

i7 = input_opener("7.txt", ",", int)
# I am very ashamed of this but sometimes you got to do what you got to do
# print(day_7(i7))
# print(day_7(i7,"2"))

