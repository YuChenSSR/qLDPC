"""Unit tests for codes.py

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
import networkx as nx
import numpy as np
import pytest

from qldpc import abstract, codes


def test_bit_codes() -> None:
    """Construction of a few classical codes."""
    assert codes.ClassicalCode.random(5, 3).num_bits == 5
    assert codes.ClassicalCode.hamming(3).get_distance() == 3

    num_bits = 5
    for code in [
        codes.ClassicalCode.repetition(num_bits, field=3),
        codes.ClassicalCode.ring(num_bits, field=4),
    ]:
        assert code.num_bits == num_bits
        assert code.dimension == 1
        assert code.get_distance() == num_bits
        assert not np.any(code.matrix @ code.get_random_word())

    # test that rank of repetition and hamming codes is independent of the field
    assert codes.ClassicalCode.repetition(3).rank == codes.ClassicalCode.repetition(3, 3).rank
    assert codes.ClassicalCode.hamming(3).rank == codes.ClassicalCode.hamming(3, 3).rank

    with pytest.raises(ValueError, match="inconsistent"):
        codes.ClassicalCode(codes.ClassicalCode.random(2, 2, field=2), field=3)


def test_dual_code(bits: int = 5, checks: int = 3, field: int = 3) -> None:
    """Dual code construction."""
    code = codes.ClassicalCode.random(bits, checks, field)
    words_a = code.words()
    words_b = code.dual().words()
    assert all(word_a @ word_b == 0 for word_a in words_a for word_b in words_b)


def test_tensor_product(
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (4, 2),
) -> None:
    """Tensor product of classical codes."""
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)
    code_ab = codes.ClassicalCode.tensor_product(code_a, code_b)
    basis = np.reshape(code_ab.generator, (-1, code_a.num_bits, code_b.num_bits))
    assert all(not (code_a.matrix @ word @ code_b.matrix.T).any() for word in basis)

    n_a, k_a, d_a = code_a.get_code_params()
    n_b, k_b, d_b = code_b.get_code_params()
    n_ab, k_ab, d_ab = code_ab.get_code_params()
    assert (n_ab, k_ab, d_ab) == (n_a * n_b, k_a * k_b, d_a * d_b)

    with pytest.raises(ValueError, match="Cannot take tensor product"):
        code_b = codes.ClassicalCode.random(*bits_checks_b, field=3)
        codes.ClassicalCode.tensor_product(code_a, code_b)


def test_classical_conversion(bits: int = 10, checks: int = 8, field: int = 3) -> None:
    """Conversion between matrix and graph representations of a classical code."""
    code = codes.ClassicalCode.random(bits, checks, field)
    graph = codes.ClassicalCode.matrix_to_graph(code.matrix)
    assert np.array_equal(code.matrix, codes.ClassicalCode.graph_to_matrix(graph))


def test_CSS_code() -> None:
    """Miscellaneous CSS code tests and coverage."""
    with pytest.raises(ValueError, match="incompatible"):
        code_x = codes.ClassicalCode.random(3, 2)
        code_z = codes.ClassicalCode.random(4, 2)
        codes.CSSCode(code_x, code_z)


def test_CSSgen_code() -> None:
    """Miscellaneous CSS code tests and coverage."""
    with pytest.raises(ValueError, match="incompatible"):
        code_x = codes.ClassicalCode.random(3, 2, 3)
        code_z = codes.ClassicalCode.random(4, 2, 3)
        codes.CSSCode_gen(code_x, code_z, 3)


@pytest.mark.parametrize("conjugate", [False, True])
def test_hyper_product(
    bits_checks_a: tuple[int, int] = (10, 8),
    bits_checks_b: tuple[int, int] = (7, 3),
    field: int = 2,
) -> None:
    """Equivalency of matrix-based and graph-based hypergraph products."""
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)

    graph_a = codes.ClassicalCode.matrix_to_graph(code_a.matrix)
    graph_b = codes.ClassicalCode.matrix_to_graph(code_b.matrix)

    code = codes.HGPCode(code_a, code_b, field)
    graph = codes.HGPCode.get_graph_product(graph_a, graph_b)
    assert np.array_equal(code.matrix, codes.QubitCode.graph_to_matrix(graph))
    assert nx.utils.graphs_equal(code.graph, graph)


def test_CSS_shifts(
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
) -> None:
    """Apply Pauli deformations to a CSS code."""
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)
    matrix_x, matrix_z, _ = codes.HGPCode.get_hyper_product(code_a, code_b)

    num_qubits = matrix_x.shape[-1]
    conjugate = tuple(qubit for qubit in range(num_qubits) if np.random.randint(2))
    shifts = {qubit: np.random.randint(3) for qubit in range(num_qubits)}
    code = codes.CSSCode(
        matrix_x,
        matrix_z,
        qubits_to_conjugate=conjugate,
        qubit_shifts=shifts,
    )

    for node_check, node_qubit, data in code.graph.edges(data=True):
        pauli_index = np.where(data[codes.Pauli].value)
        assert (code.matrix[node_check.index, pauli_index, node_qubit.index] == 1).all()


def test_CSSgen_shifts(
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
) -> None:
    """Apply Pauli deformations to a CSS code."""
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)
    matrix_x, matrix_z, _ = codes.HGPCode.get_hyper_product(code_a, code_b)

    num_qubits = matrix_x.shape[-1]
    code = codes.CSSCode_gen(matrix_x, matrix_z)
    conjugate = tuple(qubit for qubit in range(num_qubits) if np.random.randint(2))
    shifts = {qubit: np.random.randint(3) for qubit in range(num_qubits)}
    transformed_matrix = codes.CSScode_gen.conjugate(code.matrix, conjugate)
    transformed_matrix = codes.CSScode_gen.shift(transformed_matrix, shifts)

    for node_check, node_qubit, data in code.graph.edges(data=True):
        pauli_index = np.where(data[codes.Pauli].value)
        assert (transformed_matrix[node_check.index, pauli_index, node_qubit.index] == 1).all()


def test_Qudit_stabs(
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
) -> None:
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)
    matrix_x, matrix_z, _ = codes.HGPCode.get_hyper_product(code_a, code_b)
    code = codes.CSSCode_gen(matrix_x, matrix_z)
    stabs = code.matrix_to_stabilizers()
    assert len(stabs) == code.num_checks
    # for stab in stabs :
    #    print(stab)
    # print(code.matrix)
    return


def test_Qudit_graph(
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
) -> None:
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)
    matrix_x, matrix_z, _ = codes.HGPCode.get_hyper_product(code_a, code_b)
    code = codes.CSSCode_gen(matrix_x, matrix_z)
    assert np.array_equal(code.graph_to_matrix(code.graph), code.matrix)
    return


@pytest.mark.parametrize("conjugate", [False, True])
def test_trivial_lift(
    conjugate: bool,
    bits_checks_a: tuple[int, int] = (10, 8),
    bits_checks_b: tuple[int, int] = (7, 3),
    field: int = 2,
) -> None:
    """The lifted product code with a trivial lift reduces to the HGP code."""
    code_a = codes.ClassicalCode.random(*bits_checks_a, field)
    code_b = codes.ClassicalCode.random(*bits_checks_b, field)
    code_HGP = codes.HGPCode(code_a, code_b, field)

    protograph_a = abstract.TrivialGroup.to_protograph(code_a.matrix)
    protograph_b = abstract.TrivialGroup.to_protograph(code_b.matrix)
    code_LP = codes.LPCode(protograph_a, protograph_b, conjugate=conjugate)

    assert np.array_equal(code_HGP.matrix, code_LP.matrix)
    assert nx.utils.graphs_equal(code_HGP.graph, code_LP.graph)
    assert np.array_equal(code_HGP.sector_size, code_LP.sector_size)


def test_lift() -> None:
    """Verify lifting in Eqs. (8) and (10) of arXiv:2202.01702v3."""
    group = abstract.CyclicGroup(3)
    zero = abstract.Element(group)
    x0, x1, x2 = [abstract.Element(group, member) for member in group.generate()]
    base_matrix = [[x1 + x2, x0, zero], [zero, x0 + x1, x1]]

    protograph = abstract.Protograph(base_matrix)
    matrix = [
        [0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 0],
    ]
    assert np.array_equal(protograph.lift(), matrix)

    # check that the lifted product code is indeed smaller than the HGP code!
    code_HP = codes.HGPCode(matrix)
    code_LP = codes.LPCode(protograph)
    assert code_HP.num_qubits > code_LP.num_qubits
    assert code_HP.num_checks > code_LP.num_checks

    # check total number of qubits
    assert code_HP.sector_size.sum() == code_HP.num_qubits + code_HP.num_checks
    assert code_LP.sector_size.sum() == code_LP.num_qubits + code_LP.num_checks


def test_twisted_XZZX(width: int = 3) -> None:
    """Verify twisted XZZX code in Eqs.(29) and (32) of arXiv:2202.01702v3."""
    num_qubits = 2 * width**2

    # construct check matrix directly
    ring = codes.ClassicalCode.ring(width).matrix
    mat_1 = np.kron(ring, np.eye(width, dtype=int))
    mat_2 = codes.ClassicalCode.ring(num_qubits // 2).matrix
    zero_0 = np.zeros((mat_1.shape[1],) * 2, dtype=int)
    zero_1 = np.zeros((mat_1.shape[0],) * 2, dtype=int)
    zero_2 = np.zeros((mat_2.shape[0],) * 2, dtype=int)
    zero_3 = np.zeros((mat_2.shape[1],) * 2, dtype=int)
    matrix = [
        [zero_0, mat_1.T, mat_2, zero_2],
        [mat_1, zero_1, zero_3, mat_2.T],
    ]

    # construct lifted product code
    group = abstract.CyclicGroup(num_qubits // 2)
    unit = abstract.Element(group).one()
    shift = abstract.Element(group, group.generators[0])
    element_a = unit + shift**width
    element_b = unit + shift
    code = codes.LPCode([[element_a]], [[element_b]], conjugate=True)
    assert np.array_equal(np.block(matrix).ravel(), code.matrix.ravel())


def test_cyclic_codes() -> None:
    """Quasi-cyclic codes from arXiv:2308.07915."""
    dims: tuple[int, ...]

    dims = (6, 6)
    terms_a = [(0, 3), (1, 1), (1, 2)]
    terms_b = [(1, 3), (0, 1), (0, 2)]
    code = codes.QCCode(dims, terms_a, terms_b)
    assert code.num_qubits == 72
    assert code.num_logical_qubits == 12

    dims = (15, 3)
    terms_a = [(0, 9), (1, 1), (1, 2)]
    terms_b = [(0, 0), (0, 2), (0, 7)]
    code = codes.QCCode(dims, terms_a, terms_b)
    assert code.num_qubits == 90
    assert code.num_logical_qubits == 8


def test_lifted_product_codes() -> None:
    """Lifted product codes in Eq. (5) of arXiv:2308.08648."""
    for lift_dim, matrix in [
        (16, [[0, 0, 0, 0, 0], [0, 2, 4, 7, 11], [0, 3, 10, 14, 15]]),
        (21, [[0, 0, 0, 0, 0], [0, 4, 5, 7, 17], [0, 14, 18, 12, 11]]),
    ]:
        group = abstract.CyclicGroup(lift_dim)
        xx = group.generators[0]
        proto_matrix = [[abstract.Element(group, xx**power) for power in row] for row in matrix]
        protograph = abstract.Protograph(proto_matrix)
        code = codes.LPCode(protograph)
        rate = code.num_logical_qubits / code.num_qubits
        assert rate >= 2 / 17


def test_tanner_code() -> None:
    """Classical Tanner code construction.

    In order to construct a random Tanner code, we need to construct a random regular directed
    bipartite graph.  To this end, we first construct a random regular graph G = (V,E), and then
    build a directed bipartite graph G' with vertex sets (V,E).  The edges in G' have the form
    (v, {v,w}), where v is in V and {v,w} is in E.
    """
    subcode = codes.ClassicalCode.random(10, 5)
    graph = nx.random_regular_graph(subcode.num_bits, subcode.num_bits * 2 + 2)
    subgraph = nx.DiGraph()
    for edge in graph.edges:
        subgraph.add_edge(edge[0], edge)
        subgraph.add_edge(edge[1], edge)
    num_sources = sum(1 for node in subgraph if subgraph.in_degree(node) == 0)
    num_sinks = sum(1 for node in subgraph if subgraph.out_degree(node) == 0)

    # build a classical Tanner code and check that it has the right number of checks/bits
    code = codes.TannerCode(subgraph, subcode)
    assert code.num_bits == num_sinks
    assert code.num_checks == num_sources * code.subcode.num_checks


def test_surface_hgp_code() -> None:
    """The surface and toric codes as hypergraph product codes."""
    # surface code
    bit_code = codes.ClassicalCode.repetition(3)
    code = codes.HGPCode(bit_code)
    assert code.get_code_params() == (13, 1, 3)

    # toric code
    bit_code = codes.ClassicalCode.ring(3)
    code = codes.HGPCode(bit_code)
    assert code.get_code_params() == (18, 2, 3)


def test_toric_tanner_code() -> None:
    """Rotated toric code as a quantum Tanner code.

    This construction only works for cyclic groups of even order.
    """
    group = abstract.Group.product(abstract.CyclicGroup(4), repeat=2)
    shift_x, shift_y = group.generators
    subset_a = [shift_x, ~shift_x]
    subset_b = [shift_y, ~shift_y]
    subcode = codes.ClassicalCode.repetition(2)
    code = codes.QTCode(subset_a, subset_b, subcode)

    # check that this is a [[16, 2, 4]] code
    assert code.get_code_params() == (16, 2, 4)
    assert code.get_distance(lower=True) == 4
    assert code.get_distance(upper=100, ensure_nontrivial=True) == 4
    assert code.get_distance(upper=100, ensure_nontrivial=False) == 4
