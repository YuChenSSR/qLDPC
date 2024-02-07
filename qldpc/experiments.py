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

from qldpc.abstract import CyclicGroup, Group, GroupMember, SpecialLinearGroup
from qldpc.codes import ClassicalCode, QTCode

import qldpc.random_methods as generate


def reconstruct_CHcode(file, blocklength: int, hamming:int, field:int = 2):
    loaded = np.load(file)
    gen_a = loaded['gen'][0]
    gen_b = loaded['gen'][1]
    cyclegroup = CyclicGroup(blocklength)
    shift_one = cyclegroup.generators[0]   
    subset_a = [shift_one**a for a in gen_a]
    subset_b = [shift_one**b for b in gen_b]
    code_a = ClassicalCode.hamming(hamming, field)
    code_b = ~code_a
    return QTCode(subset_a, subset_b, code_a, code_b, twopartite=False)


np.set_printoptions(linewidth=200)

field = 2
hamming = 3

for blocklength in range(20,25):
    for attempt in range(10):
        file = f'./experiment_arrays/test_cycle_{blocklength}_ham{hamming}_try_{attempt}.npz'
        print(f'Testing Cyclic Codes of length {blocklength}, try_{attempt}')
        generate.random_cyclicQTcode(blocklength, field, hamming=hamming, save_file=file)
        # tannercode = reconstruct_CHcode(file, blocklength, hamming=2, field=2)
        # params = [
        # tannercode.num_qubits,
        # tannercode.dimension,
        # tannercode.get_distance(upper=5, ensure_nontrivial=False),
        # tannercode.get_weight(),]
        # print("Final code params:", params)



    


#random_linearQTcode(sl_field, hamming=3)
# if tannercode.get_distance(upper=10, ensure_nontrivial=False) > 20:
#    np.save
# print(np.any(tannercode.matrix))
""" Experiment with cyclic codes upto like 20?
Fix base codes to be Hamming[7,4] and its dual [7,3]
"""

# Z = CyclicGroup(5)
# subset_a = Z.random_symmetric_subset(3, seed=seed)
# print([p(0) for p in subset_a])

# loaded = np.load(file)
# print(loaded['params'])
# print(loaded['gen'][0])
