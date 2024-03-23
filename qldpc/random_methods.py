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


def random_basecodes(
    blocklength: int,
    field: int = 2,
) -> tuple[codes.ClassicalCode, codes.ClassicalCode]:
    """Outputs a pair of random linear codes C_A, C_B such that
    dim(C_A) + dim(C_B) = blocklength
    """
    rate = 0.4
    checks = blocklength - int((rate * blocklength))
    checks = 3
    print("Inner Code is random linear and its dual")
    code_a = codes.ClassicalCode.random(blocklength, checks, field)
    code_b = ~code_a
    print("Inner code params:")
    print(code_a.get_code_params())
    print(code_b.get_code_params())
    return code_a, code_b


def random_QTcode(
    group: abstract.Group,
    code_a: codes.ClassicalCode,
    save_file: str | None = None,
    seed: int | None = None,
) -> codes.QTCode:
    """Constructs a Quantum Tanner Code over given group using random pair of generators.
    The base codes are code_a and its dual.
    """
    code_b = ~code_a
    degree = code_a.num_bits
    subset_a = group.random_symmetric_subset(degree, seed=seed)
    subset_b = group.random_symmetric_subset(degree, seed=seed)
    tannercode = codes.QTCode(subset_a, subset_b, code_a, code_b)
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


def subset_to_array(
    subset: set[abstract.GroupMember],
) -> npt.NDArray[np.int_]:
    return np.array([s.array_form for s in subset])
