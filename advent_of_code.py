

import math
import functools
from itertools import product
import os
from pprint import pprint
import re
import traceback
import json
from typing import Dict, List, Tuple, final


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


def get_8_entries(inpt):
    entries = []
    for line in inpt:
        prefix, sample = line.split("|")
        entries.append((prefix.rstrip().split(" "), sample.lstrip().split(" ")))
    return entries

def day_8_1(inpt):
    entries = get_8_entries(inpt)
    count = 0
    for _, sample in entries:
        for digits in sample:
            if len(digits) in [2, 3, 4, 7]:
                count+=1

    return count


def set_checker(current_set: set, current_circuit: str):
    if len(current_set) == 0:
        return set(current_circuit)
    else:
        return current_set.intersection(set(current_circuit))

def day_8_2(inpt):
    entries = get_8_entries(inpt)
    count = 0

    # This is how I think of the lines as I find it hard to reason about using the original
    # values mapped to letters
    #   1
    # 2   3
    #   4
    # 5   6
    #   7

    generate_dict = lambda: ( {
        1: set(),
        2: set(),
        3: set(),
        4: set(),
        5: set(),
        6: set(),
        7: set(),
    })
    count = 0
    for prefix, sample in entries:
        mapping = generate_dict()
        for string in prefix + sample:
            if len(string) == 2:
                mapping[3] = set_checker(mapping[3], string)
                mapping[6] = set_checker(mapping[6], string)
            elif len(string) == 4:
                mapping[2] = set_checker(mapping[2], string)
                mapping[3] = set_checker(mapping[3], string)
                mapping[4] = set_checker(mapping[4], string)
                mapping[6] = set_checker(mapping[6], string)
            elif len(string) == 3:
                mapping[1] = set_checker(mapping[1], string)
                mapping[3] = set_checker(mapping[3], string)
                mapping[6] = set_checker(mapping[6], string)
            elif len(string) == 5:
                mapping[4] = set_checker(mapping[4], string)
                mapping[7] = set_checker(mapping[7], string)
            elif len(string) == 6:
                mapping[1] = set_checker(mapping[1], string)
                mapping[2] = set_checker(mapping[2], string)
                mapping[6] = set_checker(mapping[6], string)
                mapping[7] = set_checker(mapping[7], string)
            elif len(string) == 7:
                mapping[1] = set_checker(mapping[1], string)
                mapping[2] = set_checker(mapping[2], string)
                mapping[3] = set_checker(mapping[3], string)
                mapping[4] = set_checker(mapping[4], string)
                mapping[5] = set_checker(mapping[5], string)
                mapping[6] = set_checker(mapping[6], string)
                mapping[7] = set_checker(mapping[7], string)
        crack_mapping =True
        while crack_mapping:
            entered = True
            certain_entries = [next(iter(entry)) for entry in mapping.values() if len(entry) ==1]
            for k, potential_entries in mapping.items():
                if len(potential_entries) > 1:
                    for found_entry in certain_entries:
                        if found_entry in potential_entries:
                            potential_entries.remove(found_entry)
                            mapping[k]=potential_entries
            crack_mapping = False if [1,1,1,1,1,1,1] == [len(entry) for entry in mapping.values()] else True

        mapping = {next(iter(v)): str(k) for k, v in mapping.items()}
        number_sets = {
            "36": "1",

            "2346": "4",

            "136": "7",

            "1234567": "8",

            "123567": "0",
            "123467": "9",
            "124567": "6",

            "13457": "2",
            "13467": "3",
            "12467": "5",

        }
        final_number = ""
        for entry in sample:
            final_number += number_sets[
                "".join(sorted([mapping[letter] for letter in entry]))
            ]

        count += int(final_number)

    return count


i8 = input_opener("8.txt", "\n", str)
print(day_8_1(i8))
print(day_8_2(i8))


def day_9(inpt):
    mins_list = []
    bad_idxs = []
    for row_idx, row in enumerate(inpt):
        for col_idx, col_value in enumerate(row):
            is_minimum = True
            for kernel_row in (kernel_rows := list(range(row_idx-1, row_idx+2))):
                for kernel_col in (kernel_cols := list(range(col_idx-1, col_idx+2))):
                    if (
                        (row_idx, col_idx) == (kernel_row, kernel_col) or
                        (row_idx != kernel_row and col_idx != kernel_col) or
                        kernel_row < 0 or kernel_row >= len(row) or
                        kernel_col < 0 or kernel_col >= len(inpt)
                    ):
                        "do nothing"
                    else:
                        if inpt[kernel_row][kernel_col] <= col_value:
                            is_minimum = False
            if is_minimum:
                mins_list.append(col_value)
    return sum([1+minimum for minimum in mins_list])

i9 = [[int(letter) for letter in element] for element in input_opener("9.txt", "\n", str)]
print(day_9(i9))

def day_10(inpt):
    count = 0
    character_scores = {
        ")": 3,
        "]": 57,
        "}": 1197,
        ">": 25137,
    }

    for line in inpt:
        expected = []
        for character in line:
            if character == "(":
                expected.append(")")
            elif character == "[":
                expected.append("]")
            elif character == "{":
                expected.append("}")
            elif character == "<":
                expected.append(">")
            elif character != expected[-1]:
                # Uncomment for some helpful printing
                # print(f"Expected {expected[-1]}, but found {character} instead")
                count += character_scores[character]
                break
            else:
                expected.pop()
    return count

i10 =  input_opener("10.txt", "\n", str)
print(day_10(i10))

def iterate_matrix(inpt):
    for row_i, entry in enumerate(inpt):
        for col_i, value in enumerate(entry):
            yield row_i, col_i, value


def sum_octo(matrix):
    for row_i, col_i, value in iterate_matrix(matrix):
        matrix[row_i][col_i] = value + 1

def flash(matrix):
    flash_count = 0
    for row_i, col_i, value in iterate_matrix(matrix):
        if value == 10:
            flash_count += 1
            matrix[row_i][col_i] = value + 1
            # If we reached 10 we should flash and increase neighbours
            for kernel_row, kernel_col in product(
                list(range(row_i-1, row_i+2)),
                list(range(col_i-1, col_i+2))
            ):
                if len(matrix) > kernel_row >= 0 and len(matrix[0]) > kernel_col >= 0:
                    if matrix[kernel_row][kernel_col] < 10:
                        matrix[kernel_row][kernel_col] = 1 + matrix[kernel_row][kernel_col]

    return flash_count


def day_11(inpt, max_iterations = 1, part_2 = False):
    flash_count = 0
    iterations = 0
    while iterations < max_iterations:
        if part_2:
            count = 0
            for row_i, col_i, value in iterate_matrix(inpt):
                count += value
            if count == 0:
                return iterations

        iterations+=1
        sum_octo(inpt)
        attempt_flashing = True
        while attempt_flashing:
            current_flash = flash(inpt)
            flash_count += current_flash
            attempt_flashing = True if current_flash > 0 else False
        for row_i, col_i, value in iterate_matrix(inpt):
            if value == 11:
                inpt[row_i][col_i] = 0

    return flash_count

i11 =  [[int(letter) for letter in element] for element in input_opener("11.txt", "\n", str)]
print(day_11(i11, 100))
i11 =  [[int(letter) for letter in element] for element in input_opener("11.txt", "\n", str)]
print(day_11(i11, 1000, True))


def flood(graph, current_node):
    return list(product([current_node], graph.get(current_node, [])))


def generate_traversal_graph(inpt):
    graph = {}
    for entry in inpt:
        graph[entry[0]] = graph.get(entry[0], []) + [entry[1]]
        if entry[0] != "start" and entry[1] != "end":
            graph[entry[1]] = graph.get(entry[1], []) + [entry[0]]
    return graph


def can_suffix_paths(entry, suffix):
    # check if suffix has small caves already present in entry
    for cave in suffix[1:]:
        if cave in entry and re.match(r"[a-z]+", cave):
            return False

    return True


def day_12(inpt):
    last_flooding_length = 0
    graph = generate_traversal_graph(inpt)
    flooding = flood(graph, "start")

    while last_flooding_length < len(flooding):
        last_flooding_length = len(flooding)
        test = "test"
        for entry in flooding:
            subsequent_paths = flood(graph, entry[-1])
            test = test
            for suffix in subsequent_paths:
                if can_suffix_paths(entry, suffix) and (*entry, *suffix[1:]) not in flooding:
                    flooding.append((*entry, *suffix[1:]))

    finished_paths = [item for item in flooding if item[0] == "start" and item[-1] == "end"]
    return len(finished_paths)


i12 = [element.split('-') for element in input_opener("12.txt", "\n", str)]
# print(day_12(i12))