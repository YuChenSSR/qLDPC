"""Methods for generation of random codes, classical and quantum.

   Copyright 2023 The qLDPC Authors and Infleqtion Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import numpy.typing as npt

from qldpc import abstract, codes


def random_base_codes(
    blocklength: int,
) -> tuple[codes.ClassicalCode, codes.ClassicalCode]:
    """Outputs a pair of random linear codes C_A, C_B such that
    dim(C_A) + dim(C_B) = blocklength
    """
    rate = 0.4
    checks = blocklength - int((rate * blocklength))
    checks = 3
    print("Inner Code is random linear and its dual")
    code_a = codes.ClassicalCode.random(blocklength, checks)
    code_b = ~code_a
    print("Inner code params:")
    print(code_a.get_code_params())
    print(code_b.get_code_params())
    return code_a, code_b


def random_quantum_tanner(
    group: abstract.Group,
    inner_code: codes.ClassicalCode,
    save_file: str | None = None,
    seed: int | None = None,
) -> codes.QTCode:
    """Constructs a Quantum Tanner Code over given group using random pair of generators.
    The base codes are code_a and its dual.
    """
    subset_a = group.random_symmetric_subset(inner_code.num_bits, seed=seed)
    subset_b = group.random_symmetric_subset(inner_code.num_bits, seed=seed)
    code = codes.QTCode(subset_a, subset_b, inner_code, ~inner_code)
    params = code.get_code_params(bound=500)
    print("Final code params:", params)
    if save_file:
        array_a = subset_to_array(subset_a)
        array_b = subset_to_array(subset_b)
        np.savez_compressed(save_file, params=params, array_a=array_a, array_b=array_b)
    return code


def subset_to_array(
    subset: set[abstract.GroupMember],
) -> npt.NDArray[np.int_]:
    return np.array([s.array_form for s in subset])


def reconstruct_CHcode(
    blocklength: int,
    attempt: int,
    hamming: int | None = None,
    CordaroWagner: int | None = None,
    file_name: str | None = None,
) -> codes.QTCode:
    """Reconstructs the cyclic Tanner code from the file. Currently written for Hamming and Cordaro."""
    if file_name:
        file = file_name
    else:
        if hamming:
            file = f"./experiment_arrays/test_cycle_{blocklength}_ham{hamming}_try_{attempt}_2.npz"
        if CordaroWagner:
            file = f"./experiment_arrays/test_cycle_{blocklength}_Cordaro_try_{attempt}.npz"
    loaded = np.load(file)
    gen_a = loaded["gen"][0]
    gen_b = loaded["gen"][1]
    cyclegroup = abstract.CyclicGroup(blocklength)
    shift_one = cyclegroup.generators[0]
    subset_a = [shift_one**a for a in gen_a]
    subset_b = [shift_one**b for b in gen_b]
    print(list(np.sort(gen_a)))
    print(list(np.sort(gen_b)))
    code_a: codes.ClassicalCode
    if hamming:
        code_a = codes.HammingCode(hamming)
        code_b = ~code_a
    if CordaroWagner:
        code_a = codes.CordaroWagnerCode(CordaroWagner)
        code_b = ~code_a
    return codes.QTCode(subset_a, subset_b, code_a, code_b, bipartite=False)


list_prod = [(2, 2), (2, 4), (2, 2, 2), (3, 3), (2, 6), (2, 8), (4, 4), (2, 2, 2, 2), (3, 6)]
list_dihedral = [8, 10, 12, 14, 16, 18, 20]
list_dicyclic = [8, 12, 16, 20]


def modified_hamming(length: int, field: int | None = None) -> codes.ClassicalCode:
    """Modified Hammming codes."""
    base_code = codes.HammingCode(3, field=field)
    if length == 4:
        return base_code.shorten(2, 3).puncture(4)
    if length == 5:
        return base_code.shorten(2, 3)
    if length == 6:
        return base_code.shorten(3)
    raise ValueError(f"Unrecognized length for modified Hamming code: {length}")


np.set_printoptions(linewidth=200)

hamming = 3
test = False
check = False
# code_a = codes.ClassicalCode.hamming(hamming)
code_a = codes.CordaroWagnerCode(4)
# code_a = codes.ClassicalCode.RepSum(blocklength)
# group = DihedralGroup(blocklength)


def test_group_allcodes(order: int) -> None:
    list_codes = [
        (codes.HammingCode(2), "Hamming2"),
        (codes.CordaroWagnerCode(4), "Cordaro4"),
    ]
    list_groups = list(abstract.SmallGroup.generator(order))
    if order > 6:
        list_codes += [(codes.CordaroWagnerCode(5), "Cordaro5")]
    if order > 7:
        list_codes += [
            (codes.CordaroWagnerCode(6), "Cordaro6"),
            (codes.RepSumCode(6), "RepSum6"),
        ]
    if order > 9:
        list_codes += [(codes.HammingCode(3), "Hamming3")]
    num = len(list_groups)
    for ind in range(num):
        print(f"\nTesting Group of order {order} with ID {ind+1}")
        for code_a, name in list_codes:
            print(f"Testing using Base Code -- {name}")
            for attempt in range(20):
                file = f"./experiment_arrays/all_groups/test_group_{order,ind}_{name}_try_{attempt}.npz"
                random_quantum_tanner(list_groups[ind], code_a, save_file=file)


for order in range(5, 21):
    test_group_allcodes(order)


if test:
    for blocklength in list_prod[1:]:
        group = abstract.AbelianGroup(*blocklength)
        for attempt in range(20):
            file = f"./experiment_arrays/test_prodcycle_{blocklength}_Cordaro4_try_{attempt}.npz"
            print(f"Testing Product Cyclic Codes of length {blocklength}, try_{attempt}")
            random_quantum_tanner(group, code_a, save_file=file)
            # tannercode = reconstruct_CHcode(file, blocklength, hamming=2)

if check:
    block = 15
    hamming = 3
    attempt_list = [4, 13, 19]
    # attempt_list = [28]
    # file = f'./experiment_arrays/test_cycle_{blocklength}_Cordaro_try_{attempt}.npz'
    for attempt in attempt_list:
        print(f"Cordaro -- Blocklength {block}, Attempt {attempt}")
        code = reconstruct_CHcode(block, attempt, CordaroWagner=6)
        params = [
            code.num_qubits,
            code.dimension,
            code.get_distance(upper=1000, ensure_nontrivial=False),
            code.get_weight(),
        ]
        print(params)


"""Best Codes

6,2,4,2
10, 3, 24,3 [490, 34, 22, 18]
11,3,13,2 [539, 35, 34, 18]
12, 3, 4,2 [588, 44, 28, 18]
12, 3, 12, 2 [588, 36, 33, 18]
13, 3, [2, 6, 28 ] , 2 [637, 37, [32, 34, 40], 18]

"""
