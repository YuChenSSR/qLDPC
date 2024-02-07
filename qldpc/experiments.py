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


np.set_printoptions(linewidth=200)

blocklength = 7
field = 2
seed = 6
#generate.random_cyclicQTcode(blocklength, field, hamming=2, seed=seed)
#random_linearQTcode(sl_field, hamming=3)
# if tannercode.get_distance(upper=10, ensure_nontrivial=False) > 20:
#    np.save
# print(np.any(tannercode.matrix))
""" Experiment with cyclic codes upto like 20?
Fix base codes to be Hamming[7,4] and its dual [7,3]
"""

Z = CyclicGroup(5)
subset_a = Z.random_symmetric_subset(3, seed=seed)
print([p(0) for p in subset_a])