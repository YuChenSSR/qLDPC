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
import numpy.typing as npt

import qldpc.abstract
from qldpc.abstract import CyclicGroup, Group, GroupMember, SymmetricGroup
from qldpc.codes import ClassicalCode, QTCode


def generate_cyclicgroup(order: int | Sequence[int]) -> Group:
    """Generates Cyclic group of given order.
    Order: Can be an integer k --> Z_k or a tuple,
            (k_1,k_2, ... , k_r) --> Z_{k_1} x ... x Z_{k_r}
    """
    cyclegroup: CyclicGroup | Group
    if isinstance(order, int):
        cyclegroup = CyclicGroup(order)
    else:
        cyclegroup = CyclicGroup(order[0])
        for i in order[1:]:
            cyclegroup = cyclegroup * CyclicGroup(i)
    return cyclegroup


def generate_groups_of_order(order: int) -> Sequence[Group]:
    """Generates a list of all groups of given order upto 20
    List is ordered by GAP ID.
    """
    assert order > 1 and order < 21
    list_single = [2, 3, 5, 7, 11, 13, 15, 17, 19]

    if order in list_single:
        return [generate_cyclicgroup(order)]

    elif order == 4:
        return [generate_cyclicgroup(4), generate_cyclicgroup((2, 2))]

    elif order == 6:
        return [SymmetricGroup(3), generate_cyclicgroup(6)]

    elif order == 8:
        g_1 = generate_cyclicgroup(8)
        g_2 = generate_cyclicgroup((2, 4))
        g_3 = qldpc.abstract.DihedralGroup(4)
        g_4 = qldpc.abstract.DiCyclicGroup(8)
        g_5 = generate_cyclicgroup((2, 2, 2))
        return [g_1, g_2, g_3, g_4, g_5]

    elif order == 9:
        return [generate_cyclicgroup(9), generate_cyclicgroup((3, 3))]

    elif order == 10:
        return [qldpc.abstract.DihedralGroup(5), generate_cyclicgroup(10)]

    elif order == 12:
        h_1 = qldpc.abstract.DiCyclicGroup(12)
        h_2 = generate_cyclicgroup(12)
        h_3 = qldpc.abstract.AlternatingGroup(4)
        h_4 = qldpc.abstract.DihedralGroup(6)
        h_5 = generate_cyclicgroup((2, 6))
        return [h_1, h_2, h_3, h_4, h_5]

    elif order == 14:
        return [qldpc.abstract.DihedralGroup(7), generate_cyclicgroup(14)]

    elif order == 16:
        return [qldpc.abstract.Order16(i) for i in range(1, 15)]

    elif order == 18:
        return [qldpc.abstract.Order18(i) for i in range(1, 6)]

    elif order == 20:
        return [qldpc.abstract.Order20(i) for i in range(1, 6)]


# def random_cyclicgens(
#     order: int | Sequence[int], degree: int, seed: int | None = None
# ) -> tuple[Group, set[GroupMember], set[GroupMember], npt.NDArray[np.int_]| None]:
#     """Generates a pair of random subsets of a cyclic group or a product of cyclic groups.
#     Order: Can be an integer k --> Z_k or a tuple,
#             (k_1,k_2, ... , k_r) --> Z_{k_1} x ... x Z_{k_r}
#     degree: The size of the subsets (both equal to degree)
#     """
#     cyclegroup: CyclicGroup | Group
#     cyclegroup = generate_cyclicgroup(order)
#     subset_a = cyclegroup.random_symmetric_subset(degree, seed=seed)
#     subset_b = cyclegroup.random_symmetric_subset(degree, seed=seed)
#     print(f"Quantum Tanner Code over Cyclic group of order {order} with {degree} generators")
#     if isinstance(order, int):
#         # print("Generators")
#         generators = np.array([[p(0) for p in subset_a], [p(0) for p in subset_b]])
#         # print(generators)
#     else:
#         generators = None
#     return cyclegroup, subset_a, subset_b, generators


def random_basecodes(
    blocklength: int,
    field: int = 2,
) -> tuple[ClassicalCode, ClassicalCode]:
    """Outputs a pair of random linear codes C_A, C_B such that
    dim(C_A) + dim(C_B) = blocklength
    """
    rate = 0.4
    checks = blocklength - int((rate * blocklength))
    checks = 3
    print("Inner Code is random linear and its dual")
    code_a = ClassicalCode.random(blocklength, checks, field)
    code_b = ~code_a
    print("Inner code params:")
    print(code_a.get_code_params())
    print(code_b.get_code_params())
    return code_a, code_b


def subset_to_array(
    subset: set[GroupMember],
) -> npt.NDArray[np.int_]:
    return np.array([s.array_form for s in subset])


def random_QTcode(
    group: Group,
    code_a: ClassicalCode,
    save_file: str | None = None,
    seed: int | None = None,
) -> QTCode:
    """Constructs a Quantum Tanner Code over given group using random pair of generators.
    The base codes are code_a and its dual.
    """
    code_b = ~code_a
    degree = code_a.num_bits
    subset_a = group.random_symmetric_subset(degree, seed=seed)
    subset_b = group.random_symmetric_subset(degree, seed=seed)
    tannercode = QTCode(subset_a, subset_b, code_a, code_b)
    params = [
        tannercode.num_qubits,
        tannercode.dimension,
        tannercode.get_distance(upper=500, ensure_nontrivial=False),
        tannercode.get_weight(),
    ]
    print("Final code params:", params)
    if save_file:
        array_a = subset_to_array(subset_a)
        array_b = subset_to_array(subset_b)
        np.savez_compressed(save_file, params=params, array_a=array_a, array_b=array_b)
    return tannercode
