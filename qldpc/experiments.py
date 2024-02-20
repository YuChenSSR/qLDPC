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

# import itertools
# import time
# import galois

from collections.abc import Sequence

import numpy as np

import qldpc.random_methods as generate
from qldpc.abstract import CyclicGroup
from qldpc.codes import ClassicalCode, QTCode


def reconstruct_CHcode(
    blocklength: int,
    attempt: int,
    hamming: int | None = None,
    CordaroWagner: int | None = None,
    file_name=None,
    field: int = 2,
) -> QTCode:
    """ Reconstructs the cyclic Tanner code from the file. Currently written for Hamming and Cordaro. 
"""
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
    cyclegroup = CyclicGroup(blocklength)
    shift_one = cyclegroup.generators[0]
    subset_a = [shift_one**a for a in gen_a]
    subset_b = [shift_one**b for b in gen_b]
    print(list(np.sort(gen_a)))
    print(list(np.sort(gen_b)))
    if hamming:
        code_a = ClassicalCode.hamming(hamming, field)
        code_b = ~code_a
    if CordaroWagner:
        code_a = ClassicalCode.CordaroWagner(CordaroWagner, field=2)
        code_b = ~code_a
    return QTCode(subset_a, subset_b, code_a, code_b, twopartite=False)


np.set_printoptions(linewidth=200)

field = 2
hamming = 3
test = True
check = False

if test:
    for blocklength in range(9, 14):
        for attempt in range(3):
            # file = f'./experiment_arrays/test_cycle_{blocklength}_RepSum_try_{attempt}.npz'
            print(f"Testing Cyclic Codes of length {blocklength}, try_{attempt}")
            # generate.random_cyclicQTcode(blocklength, field, hamming=hamming, save_file=file)
            generate.random_cyclicQTcode(blocklength, field)
            # tannercode = reconstruct_CHcode(file, blocklength, hamming=2, field=2)
            # params = [
            # tannercode.num_qubits,
            # tannercode.dimension,
            # tannercode.get_distance(upper=100, ensure_nontrivial=False),
            # tannercode.get_weight(),]
            # print("Final code params:", params)

if check:
    blocklength = 15
    hamming = 3
    attempt_list = [4, 13, 19]
    # attempt_list = [28]
    # file = f'./experiment_arrays/test_cycle_{blocklength}_Cordaro_try_{attempt}.npz'
    for attempt in attempt_list:
        print(f"Cordaro -- Blocklength {blocklength}, Attempt {attempt}")
        code = reconstruct_CHcode(blocklength, attempt, CordaroWagner=6)
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
