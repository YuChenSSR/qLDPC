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


def random_cyclicgens(
    order: int | Sequence[int], degree: int, seed: int | None = None
) -> tuple[Group, set[GroupMember], set[GroupMember]]:
    """Generates a pair of random subsets of a cyclic group or a product of cyclic groups.
    Order: Can be an integer k --> Z_k or a tuple,
            (k_1,k_2, ... , k_r) --> Z_{k_1} x ... x Z_{k_r}
    degree: The size of the subsets (both equal to degree)
    """
    cyclegroup: CyclicGroup | Group
    if isinstance(order, int):
        cyclegroup = CyclicGroup(order)
    else:
        cyclegroup = CyclicGroup(order[0])
        for i in order[1:]:
            cyclegroup = cyclegroup * CyclicGroup(i)
    subset_a = cyclegroup.random_symmetric_subset(degree, seed=seed)
    subset_b = cyclegroup.random_symmetric_subset(degree, seed=seed)
    print(f"Quantum Tanner Code over Cyclic group of order {order} with {degree} generators")
    if isinstance(order, int):
        # print("Generators")
        generators = np.array([[p(0) for p in subset_a], [p(0) for p in subset_b]])
        # print(generators)
    return cyclegroup, subset_a, subset_b, generators


def random_lineargens(
    sl_field: int, degree: int, dimension: int = 2
) -> tuple[SpecialLinearGroup, set[GroupMember], set[GroupMember]]:
    """Generates a pair of random subsets of SL(dimension, sl_field) of size degree."""
    lineargroup = SpecialLinearGroup(sl_field, dimension)
    subset_a = lineargroup.random_symmetric_subset(degree)
    subset_b = lineargroup.random_symmetric_subset(degree)
    print(f"Quantum Tanner Code over SL({sl_field}, {dimension}) with {degree} generators ")
    return lineargroup, subset_a, subset_b


def random_basecodes(
    blocklength: int,
    field: int = 2,
    hamming: int | None = None,
    CordaroWagner: bool = False,
    RepSum: bool = False,
    save_file: str | None = None,
) -> tuple[ClassicalCode, ClassicalCode]:
    """Outputs a pair of codes C_A, C_B such that
    dim(C_A) + dim(C_B) = blocklength
    """
    if hamming is not None:
        assert blocklength == 2**hamming - 1
        print(f"Inner Code is Hamming and its dual of rank {hamming}")
        code_a = ClassicalCode.hamming(hamming, field)
        code_b = ~code_a
    elif CordaroWagner:
        assert blocklength in [5, 6]
        code_a = ClassicalCode.CordaroWagner(blocklength, field=2)
        code_b = ~code_a
    elif RepSum:
        assert blocklength in [5, 6]
        code_a = ClassicalCode.RepSum(blocklength, field=2)
        code_b = ~code_a
    else:
        rate = 0.4
        checks =  blocklength - int((rate * blocklength))
        checks = 3
        print("Inner Code is random linear and its dual")
        code_a = ClassicalCode.random(blocklength, checks, field)
        code_b = ~code_a
    print("Inner code params:")
    print(code_a.get_code_params())
    print(code_b.get_code_params())
    return code_a, code_b


def random_cyclicQTcode(
    order: int | Sequence[int],
    field: int = 2,
    hamming: int | None = None,
    CordaroWagner: bool = False,
    RepSum: bool = False,
    save_file: str | None = None,
    seed: int | None = None,
) -> QTCode:
    """Constructs a Quantum Tanner Code over Cyclic group of given order
    with random generators.
    """
    if isinstance(order, int):
        size = order
    else:
        size = np.prod(np.array(order))

    if hamming:
        deg = 2**hamming - 1
        assert deg <= size

    elif CordaroWagner or RepSum:
        deg = 6

    else:
        deg = 6
    _, subset_a, subset_b, generators = random_cyclicgens(order, deg, seed=seed)
    code_a, code_b = random_basecodes(
        deg, field, hamming=hamming, CordaroWagner=CordaroWagner, RepSum=RepSum, save_file=save_file
    )
    # code_a = ClassicalCode.repetition(deg)
    # code_b = ~code_a
    tannercode = QTCode(subset_a, subset_b, code_a, code_b, twopartite=False)
    params = [
        tannercode.num_qubits,
        tannercode.dimension,
        tannercode.get_distance(upper=500, ensure_nontrivial=False),
        tannercode.get_weight(),
    ]
    print("Final code params:", params)
    if save_file:
        if hamming or CordaroWagner or RepSum:
            np.savez_compressed(save_file, params=params, gen=generators)
        else:
            np.savez_compressed(
                save_file, params=params, gen=generators, code_a=code_a.matrix, code_b=code_b.matrix
            )
    return tannercode


def random_linearQTcode(
    sl_field: int,
    field: int = 2,
    dimension: int = 2,
    hamming: int | None = None,
    save_file: str | None = None,
) -> QTCode:
    """Constructs a Quantum Tanner Code over SpecialLinear group of given order
    with random generators.
    """
    if hamming:
        deg = 2**hamming - 1
    else:
        deg = 5
    _, subset_a, subset_b = random_lineargens(sl_field, deg, dimension)
    code_a, code_b = random_basecodes(deg, field, hamming=hamming, save_file=save_file)
    tannercode = QTCode(subset_a, subset_b, code_a, code_b)
    params = [
        tannercode.num_qubits,
        tannercode.dimension,
        tannercode.get_distance(upper=5, ensure_nontrivial=False),
        tannercode.get_weight(),
    ]
    print("Final code params:", params)
    return tannercode
