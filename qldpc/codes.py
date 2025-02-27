"""Error correction code constructions

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

from __future__ import annotations

import abc
import functools
import itertools
import random
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Literal

import galois
import ldpc.mod2
import networkx as nx
import numpy as np
import numpy.typing as npt
import sympy
import sympy.combinatorics as comb

import qldpc
from qldpc import abstract, named_codes
from qldpc.objects import (
    PAULIS_XZ,
    CayleyComplex,
    ChainComplex,
    Node,
    Pauli,
    PauliXZ,
    QuditOperator,
)

DEFAULT_FIELD_ORDER = 2


def get_random_nontrivial_vec(field: type[galois.FieldArray], size: int) -> galois.FieldArray:
    """Get a random nontrivial vector of a given size."""
    while not (vec := field.Random(size)).any():
        pass  # pragma: no cover
    return vec


################################################################################
# template error correction code class


# TODO(?): support sparse parity check matrices
class AbstractCode(abc.ABC):
    """Template class for error-correcting codes."""

    _field: type[galois.FieldArray]

    def __init__(
        self,
        matrix: AbstractCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
    ) -> None:
        """Construct a code from a parity check matrix over a finite field.

        The base field is taken to be F_2 by default.
        """
        self._matrix: galois.FieldArray
        if isinstance(matrix, type(self)):
            self._field = matrix.field
            self._matrix = matrix.matrix
        elif isinstance(matrix, galois.FieldArray):
            self._field = type(matrix)
            self._matrix = matrix
        else:
            self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
            self._matrix = self.field(np.array(matrix))

        if field is not None and field != self.field.order:
            raise ValueError(
                f"Field argument {field} is inconsistent with the given code, which is defined"
                f" over F_{self.field.order}"
            )

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field over which this code is defined."""
        return self._field

    @property
    def matrix(self) -> galois.FieldArray:
        """Parity check matrix of this code."""
        return self._matrix

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Tanner graph of this code."""
        return self.matrix_to_graph(self.matrix)

    @classmethod
    @abc.abstractmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""

    @classmethod
    @abc.abstractmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""


################################################################################
# classical codes


# TODO:
# - add code concatenation
class ClassicalCode(AbstractCode):
    """Classical linear error-correcting code over a finite field F_q.

    A classical binary code C = {x} is a set of vectors x (with entries in F_q) called code words.
    We consider only linear codes, for which any linear combination of code words is also code word.

    Operationally, we define a classical code by a parity check matrix H with dimensions
    (num_checks, num_bits).  Each row of H represents a linear constraint (a "check") that code
    words must satisfy.  A vector x is a code word iff H @ x = 0.
    """

    _matrix: galois.FieldArray
    _exact_distance: int | None = None

    def __contains__(
        self, words: npt.NDArray[np.int_] | Sequence[int] | Sequence[Sequence[int]] | ClassicalCode
    ) -> bool:
        """Does this code contain the given word(s)?

        If passed a ClassicalCode for "words", interpret it to mean "all words in the given code",
        which are spanned by the code's generator matrix.
        """
        if isinstance(words, ClassicalCode):
            words = words.generator
        return not np.any(self.matrix @ self.field(words).T)

    @classmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix H into a Tanner graph.

        The Tanner graph is a bipartite graph with (num_checks, num_bits) vertices, respectively
        identified with the checks and bits of the code.  The check vertex c and the bit vertex b
        share an edge iff c addresses b; that is, edge (c, b) is in the graph iff H[c, b] != 0.
        """
        graph = nx.DiGraph()
        for row, col in zip(*np.nonzero(matrix)):
            node_c = Node(index=int(row), is_data=False)
            node_d = Node(index=int(col), is_data=True)
            graph.add_edge(node_c, node_d, val=matrix[row][col])
        if isinstance(matrix, galois.FieldArray):
            graph.order = type(matrix).order
        return graph

    @classmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""
        num_bits = sum(1 for node in graph.nodes() if node.is_data)
        num_checks = len(graph.nodes()) - num_bits
        field = getattr(graph, "order", DEFAULT_FIELD_ORDER)
        matrix = galois.GF(field).Zeros((num_checks, num_bits))
        for node_c, node_b, data in graph.edges(data=True):
            matrix[node_c.index, node_b.index] = data.get("val", 1)
        return matrix

    @functools.cached_property
    def generator(self) -> galois.FieldArray:
        """Generator of this code: a matrix whose rows for a basis for code words."""
        return self.matrix.null_space()

    def __eq__(self, other: object) -> bool:
        """Equality test between two classical code instances."""
        return (
            isinstance(other, ClassicalCode)
            and self.field is other.field
            and np.array_equal(self.matrix, other.matrix)
        )

    @classmethod
    def equiv(cls, code_a: ClassicalCode, code_b: ClassicalCode) -> bool:
        """Test equivalence between two classical codes.

        Two classical codes are equivalent if they have the same code words.  Equivalently, codes
        C_a and C_b are equivalent if they contain each other, C_a ⊆ C_b and C_b ⊆ C_a.
        """
        return code_a.field is code_b.field and code_a in code_b and code_b in code_a

    def words(self) -> galois.FieldArray:
        """Code words of this code."""
        vectors = itertools.product(self.field.elements, repeat=self.generator.shape[0])
        return self.field(list(vectors)) @ self.generator

    def get_random_word(self) -> galois.FieldArray:
        """Random code word: a sum all generators with random field coefficients."""
        return self.field.Random(self.generator.shape[0]) @ self.generator

    def dual(self) -> ClassicalCode:
        """Dual to this code.

        The dual code ~C is the set of bitstrings orthogonal to C:
        ~C = { x : x @ y = 0 for all y in C }.
        The parity check matrix of ~C is equal to the generator of C.
        """
        return ClassicalCode(self.generator)

    def __invert__(self) -> ClassicalCode:
        return self.dual()

    @classmethod
    def tensor_product(cls, code_a: ClassicalCode, code_b: ClassicalCode) -> ClassicalCode:
        """Tensor product C_a ⨂ C_b of two codes C_a and C_b.

        Let G_a and G_b respectively denote the generators C_a and C_b.
        Definition: C_a ⨂ C_b is the code whose generators are G_a ⨂ G_b.

        Observation: G_a ⨂ G_b is the check matrix of ~(C_a ⨂ C_b).
        We therefore construct ~(C_a ⨂ C_b) and return its dual ~~(C_a ⨂ C_b) = C_a ⨂ C_b.
        """
        if code_a.field is not code_b.field:
            raise ValueError("Cannot take tensor product of codes over different fields")
        gen_a: npt.NDArray[np.int_] = code_a.generator
        gen_b: npt.NDArray[np.int_] = code_b.generator
        return ~ClassicalCode(np.kron(gen_a, gen_b))

    @property
    def num_checks(self) -> int:
        """Number of check bits in this code."""
        return self._matrix.shape[0]

    @property
    def num_bits(self) -> int:
        """Number of data bits in this code."""
        return self._matrix.shape[1]

    @functools.cached_property
    def rank(self) -> int:
        """Rank of this code's parity check matrix.

        Equivalently, the number of linearly independent parity checks in this code.
        """
        if self.field.order == 2:
            return ldpc.mod2.rank(self._matrix)
        return np.linalg.matrix_rank(self._matrix)

    @property
    def dimension(self) -> int:
        """The number of logical bits encoded by this code."""
        return self.num_bits - self.rank

    def get_distance(
        self,
        *,
        bound: int | bool | None = None,
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: object,
    ) -> int:
        """Compute (or upper bound) the minimal weight of a nontrivial code word.

        If passed a vector, compute the minimal Hamming distance between the vector and a code word.

        Additional arguments, if applicable, are passed to a decoder in
        `ClassicalCode.get_one_distance_bound`.
        """
        if bound is None:
            return self.get_distance_exact(vector=vector)
        return self.get_distance_bound(num_trials=int(bound), vector=vector, **decoder_args)

    def get_distance_exact(
        self, *, vector: Sequence[int] | npt.NDArray[np.int_] | None = None
    ) -> int:
        """Compute the minimal weight of a nontrivial code word by brute force.

        If passed a vector, compute the minimal Hamming distance between the vector and a code word.
        """
        if vector is not None:
            words = self.words() - self.field(vector)[np.newaxis, :]
            return np.min(np.count_nonzero(words.view(np.ndarray), axis=1))

        # if we know the exact code distance, return it
        if self._exact_distance is not None:
            return self._exact_distance

        # we do not know the exact distance, so compute it
        words = self.words()[1:]
        self._exact_distance = np.min(np.count_nonzero(words.view(np.ndarray), axis=1))
        return self._exact_distance

    def get_distance_bound(
        self,
        num_trials: int = 1,
        *,
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: object,
    ) -> int:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        If passed a vector, compute the minimal Hamming distance between the vector and a code word.

        Additional arguments, if applicable, are passed to a decoder in
        `ClassicalCode.get_one_distance_bound`.
        """
        distance_bounds = (
            self.get_one_distance_bound(vector=vector, **decoder_args) for _ in range(num_trials)
        )
        return min(distance_bounds, default=self.num_bits)

    def get_one_distance_bound(
        self, *, vector: Sequence[int] | npt.NDArray[np.int_] | None = None, **decoder_args: object
    ) -> int:
        """Compute a single upper bound on code distance.

        The code distance is the minimal Hamming distance between two code words, or equivalently
        the minimal Hamming weight of a nonzero code word.  To find a minimal nonzero code word we
        decode a trivial (all-0) syndrome, but enforce that the code word has nonzero overlap with a
        random word, which excludes the all-0 word as a candidate.

        If passed a vector, compute the minimal Hamming distance between the vector and a code word.
        Equivalently, we can interpret the given vector as an error, and find a minimal-weight
        correction from decoding the syndrome induced by this vector.

        Additional arguments, if applicable, are passed to a decoder.
        """
        if vector is not None:
            # find the distance of the given vector from a code word
            correction = qldpc.decoder.decode(
                self.matrix,
                self.matrix @ self.field(vector),
                **decoder_args,
            )
            return int(np.count_nonzero(correction))

        # effective syndrome: a trivial "actual" syndrome, and a nonzero overlap with a random word
        effective_syndrome = np.zeros(self.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1
        _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field, bound_index=-1)

        valid_candidate_found = False
        while not valid_candidate_found:
            # construct the effective check matrix
            random_word = get_random_nontrivial_vec(self.field, self.num_bits)
            effective_check_matrix = np.vstack([self.matrix, random_word]).view(np.ndarray)

            # find a low-weight candidate code word
            candidate = qldpc.decoder.decode(
                effective_check_matrix,
                effective_syndrome,
                **decoder_args,
            )

            # check whether we found a valid candidate
            actual_syndrome = effective_check_matrix @ candidate % self.field.order
            valid_candidate_found = np.array_equal(actual_syndrome, effective_syndrome)

        return int(np.count_nonzero(candidate))

    def get_code_params(
        self, *, bound: int | bool | None = None, **decoder_args: object
    ) -> tuple[int, int, int, int]:
        """Compute the parameters of this code: [n,k,d,w].

        Here:
        - n is the number of data bits
        - k is the number of encoded ("logical") bits
        - d is the code distance
        - w is the maximal weight of (i.e., number of bits addressed by) a parity check

        Keyword arguments are passed to the calculation of code distance.
        """
        distance = self.get_distance(bound=bound, vector=None, **decoder_args)
        return self.num_bits, self.dimension, distance, self.get_weight()

    def get_weight(self) -> int:
        """Compute the weight of the largest check."""
        return max(np.count_nonzero(row) for row in self.matrix)

    @classmethod
    def random(cls, bits: int, checks: int, field: int | None = None) -> ClassicalCode:
        """Construct a random classical code with the given number of bits and nontrivial checks.

        Reject parity check matrices that have a row or column of all zeroes.
        """
        code_field = galois.GF(field or DEFAULT_FIELD_ORDER)

        def has_zero_row_or_column(matrix: galois.FieldArray) -> bool:
            """Does the given matrix have a row or column that is all zeroes?"""
            return any(not row.any() for row in matrix) or any(not col.any() for col in matrix.T)

        while has_zero_row_or_column(matrix := code_field.Random((checks, bits))):
            pass

        return ClassicalCode(matrix)

    @classmethod
    def from_generator(
        self, generator: npt.NDArray[np.int_] | Sequence[Sequence[int]], field: int | None = None
    ) -> ClassicalCode:
        """Construct a ClassicalCode from a generator matrix."""
        return ~ClassicalCode(generator, field)

    @classmethod
    def from_name(cls, name: str) -> ClassicalCode:
        """Named code in the GAP computer algebra system."""
        standardized_name = name.strip().replace(" ", "")  # remove whitespace
        matrix, field = named_codes.get_code(standardized_name)
        return ClassicalCode(matrix, field)

    def puncture(self, *bits: int) -> ClassicalCode:
        """Delete the specified bits from a code.

        To delete bits from the code, we remove the corresponding columns from its generator matrix.
        """
        assert all(0 <= bit < self.num_bits for bit in bits)
        bits_to_keep = [bit for bit in range(self.num_bits) if bit not in bits]
        generator = [word[bits_to_keep] for word in self.generator]
        return ClassicalCode.from_generator(generator, self.field.order)

    def shorten(self, *bits: int) -> ClassicalCode:
        """Shorten a code to the words that are zero on the specified bits, and delete those bits.

        To shorten a code on a given bit, we:
        - move the bit to the first position,
        - row-reduce the generator matrix into the form [ identity_matrix, other_stuff ], and
        - delete the first row and column from the generator matrix.
        """
        assert all(0 <= bit < self.num_bits for bit in bits)
        generator = self.generator
        for bit in sorted(bits, reverse=True):
            generator = np.roll(generator, -bit, axis=1)  # type:ignore[assignment]
            generator = generator.row_reduce()[1:, 1:]
            generator = np.roll(generator, bit, axis=1)  # type:ignore[assignment]
        return ClassicalCode.from_generator(generator)


class RepetitionCode(ClassicalCode):
    """Classical repetition code."""

    def __init__(self, bits: int, field: int | None = None) -> None:
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._matrix = self.field.Zeros((bits - 1, bits))
        for row in range(bits - 1):
            self._matrix[row, row] = 1
            self._matrix[row, row + 1] = -self.field(1)


class RingCode(ClassicalCode):
    """Classical ring code: repetition code with periodic boundary conditions."""

    def __init__(self, bits: int, field: int | None = None) -> None:
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._matrix = self.field.Zeros((bits, bits))
        for row in range(bits):
            self._matrix[row, row] = 1
            self._matrix[row, (row + 1) % bits] = -self.field(1)


class HammingCode(ClassicalCode):
    """Classical Hamming code."""

    def __init__(self, rank: int, field: int | None = None) -> None:
        """Construct a Hamming code of a given rank."""
        self._exact_distance = 3
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        if self.field.order == 2:
            # parity check matrix: columns = all nonzero bitstrings
            bitstrings = list(itertools.product([0, 1], repeat=rank))
            self._matrix = self.field(bitstrings[1:]).T

        else:
            # More generally, columns = [maximal set of linearly independent strings], so collect
            # together all strings whose first nonzero element is a 1.
            strings = [
                (0,) * top_row + (1,) + rest
                for top_row in range(rank - 1, -1, -1)
                for rest in itertools.product(range(self.field.order), repeat=rank - top_row - 1)
            ]
            self._matrix = self.field(strings).T


class ReedSolomonCode(ClassicalCode):
    """Classical Reed-Solomon code.

    Source: https://galois.readthedocs.io/en/v0.3.8/api/galois.ReedSolomon/
    References:
    - https://errorcorrectionzoo.org/c/reed_solomon
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(self, bits: int, dimension: int) -> None:
        ClassicalCode.__init__(self, galois.ReedSolomon(bits, dimension).H)


class BCHCode(ClassicalCode):
    """Classical binary BCH code.

    Source: https://galois.readthedocs.io/en/v0.3.8/api/galois.BCH/
    References:
    - https://errorcorrectionzoo.org/c/bch
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(self, bits: int, dimension: int) -> None:
        if "0" in format(bits, "b"):
            raise ValueError("BCH codes only defined for 2^m - 1 bits with integer m.")
        ClassicalCode.__init__(self, galois.BCH(bits, dimension).H)


class ReedMullerCode(ClassicalCode):
    """Classical Reed-Muller code.

    References:
    - https://errorcorrectionzoo.org/c/reed_muller
    - https://feog.github.io/10-coding.pdf
    """

    def __init__(self, order: int, size: int, field: int | None = None) -> None:
        self._assert_valid_params(order, size)
        self._exact_distance = 2 ** (size - order)
        self._order = order
        self._size = size

        generator = ReedMullerCode.get_generator(order, size)
        self._matrix = ClassicalCode(generator, field).generator
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)

    @classmethod
    def get_generator(cls, order: int, size: int) -> npt.NDArray[np.int_]:
        """Get the generator matrix for the specified Reed-Muller code."""
        cls._assert_valid_params(order, size)

        if order == 0:
            return np.ones(2**size, dtype=int)
        if order == size:
            return np.identity(2**size, dtype=int)

        mat_a = cls.get_generator(order, size - 1)
        mat_b = cls.get_generator(order - 1, size - 1)
        mat_z = np.zeros_like(mat_b)
        return np.block([[mat_a, mat_a], [mat_z, mat_b]]).astype(int)

    @classmethod
    def _assert_valid_params(self, order: int, size: int) -> None:
        if not (size >= 0 and 0 <= order <= size):
            raise ValueError(
                "Reed-Muller code R(r,m) must have m >= 0 and 0 <= r <= m\n"
                + f"Provided: (r,m) = ({order},{size})"
            )


################################################################################
# quantum codes


# TODO:
# - add code concatenation
# - investigate weight reduction: https://arxiv.org/abs/2402.05228
# - add is_CSS method to figure out whether this is a CSS Code
#   - see https://quantumcomputing.stackexchange.com/questions/15432/
#   - also compute and store sub-codes, if CSS
#   - also add QuditCode.to_CSS() -> CSSCode
class QuditCode(AbstractCode):
    """Quantum stabilizer code for Galois qudits, with dimension q = p^m for prime p and integer m.

    The parity check matrix of a QuditCode has dimensions (num_checks, 2 * num_qudits), and can be
    written as a block matrix in the form H = [H_x|H_z].  Each block has num_qudits columns.

    The entries H_x[c, d] = r_x and H_z[c, d] = r_z iff check c addresses qudit d with the operator
    X(r_x) * Z(r_z), where r_x, r_z range over the base field, and X(r), Z(r) are generalized Pauli
    operators.  Specifically:
    - X(r) = sum_{j=0}^{q-1} |j+r><j| is a shift operator, and
    - Z(r) = sum_{j=0}^{q-1} w^{j r} |j><j| is a phase operator, with w = exp(2 pi i / q).

    Warning: here j, r, s, etc. not integers, but elements of the Galois field GF(q), which has
    different rules for addition and multiplication when q is not a prime number.

    Helpful lecture by Gottesman: https://www.youtube.com/watch?v=JWg4zrNAF-g
    """

    _matrix: galois.FieldArray
    _exact_distance_x: int | None = None
    _exact_distance_z: int | None = None

    @property
    def num_checks(self) -> int:
        """Number of parity checks (stabilizers) in this code."""
        return self.matrix.shape[0]

    @property
    def num_qudits(self) -> int:
        """Number of data qudits in this code."""
        return self.matrix.shape[1] // 2

    @property
    def num_qubits(self) -> int:
        """Number of data qubits in this code."""
        self._assert_qubit_code()
        return self.num_qudits

    def _assert_qubit_code(self) -> None:
        if self.field.order != 2:
            raise ValueError("Attempted to call a qubit-only method with a non-qubit code")

    def get_weight(self) -> int:
        """Compute the weight of the largest check."""
        matrix_x = self.matrix[:, : self.num_qudits].view(np.ndarray)
        matrix_z = self.matrix[:, self.num_qudits :].view(np.ndarray)
        matrix = matrix_x + matrix_z  # nonzero wherever a check addresses a qudit
        return max(np.count_nonzero(row) for row in matrix)

    @classmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""
        graph = nx.DiGraph()
        matrix = np.reshape(matrix, (len(matrix), 2, -1))
        for row, col_xz, col in zip(*np.nonzero(matrix)):
            node_check = Node(index=int(row), is_data=False)
            node_qudit = Node(index=int(col), is_data=True)
            graph.add_edge(node_check, node_qudit)

            qudit_op = graph[node_check][node_qudit].get(QuditOperator, QuditOperator())
            vals_xz = list(qudit_op.value)
            vals_xz[col_xz] += int(matrix[row, col_xz, col])
            graph[node_check][node_qudit][QuditOperator] = QuditOperator(tuple(vals_xz))

        # remember order of the field, and use Pauli operators if appropriate
        if isinstance(matrix, galois.FieldArray):
            graph.order = type(matrix).order
            if graph.order == 2:
                for _, __, data in graph.edges(data=True):
                    data[Pauli] = Pauli(data[QuditOperator].value)
                    del data[QuditOperator]

        return graph

    @classmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""
        num_qudits = sum(1 for node in graph.nodes() if node.is_data)
        num_checks = len(graph.nodes()) - num_qudits
        matrix = np.zeros((num_checks, 2, num_qudits), dtype=int)
        for node_check, node_qudit, data in graph.edges(data=True):
            op = data.get(QuditOperator) or data.get(Pauli)
            matrix[node_check.index, :, node_qudit.index] = op.value
        field = graph.order if hasattr(graph, "order") else DEFAULT_FIELD_ORDER
        return galois.GF(field)(matrix.reshape(num_checks, 2 * num_qudits))

    def get_stabilizers(self) -> list[str]:
        """Stabilizers (checks) of this code, represented by strings."""
        matrix = self.matrix.reshape(self.num_checks, 2, self.num_qudits)
        stabilizers = []
        for check in range(self.num_checks):
            ops = []
            for qudit in range(self.num_qudits):
                val_x = matrix[check, Pauli.X, qudit]
                val_z = matrix[check, Pauli.Z, qudit]
                vals_xz = (val_x, val_z)
                if self.field.order == 2:
                    ops.append(str(Pauli(vals_xz)))
                else:
                    ops.append(str(QuditOperator(vals_xz)))
            stabilizers.append(" ".join(ops))
        return stabilizers

    @classmethod
    def from_stabilizers(cls, stabilizers: Iterable[str], field: int | None = None) -> QuditCode:
        """Construct a QuditCode from the provided stabilizers."""
        field = field or DEFAULT_FIELD_ORDER
        check_ops = [stabilizer.split() for stabilizer in stabilizers]
        num_checks = len(check_ops)
        num_qudits = len(check_ops[0])
        operator: type[Pauli] | type[QuditOperator] = Pauli if field == 2 else QuditOperator

        matrix = np.zeros((num_checks, 2, num_qudits), dtype=int)
        for check, check_op in enumerate(check_ops):
            if len(check_op) != num_qudits:
                raise ValueError(f"Stabilizers 0 and {check} have different lengths")
            for qudit, op in enumerate(check_op):
                matrix[check, :, qudit] = operator.from_string(op).value

        return QuditCode(matrix.reshape(num_checks, 2 * num_qudits), field)

    # TODO: generalize to any local Clifford deformation
    #       see https://arxiv.org/abs/quant-ph/0408190
    @classmethod
    def conjugate(
        cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]], qudits: slice | Sequence[int]
    ) -> npt.NDArray[np.int_]:
        """Apply local Fourier transforms to the given qudits.

        This is equivalent to swapping X-type and Z-type operators."""
        num_checks = len(matrix)
        matrix = np.reshape(matrix, (num_checks, 2, -1))
        matrix[:, :, qudits] = np.roll(matrix[:, :, qudits], 1, axis=1)
        return matrix.reshape(num_checks, -1)


class CSSCode(QuditCode):
    """CSS qudit code, with separate X-type and Z-type parity checks.

    In order for the X-type and Z-type parity checks to be "compatible", the X-type stabilizers must
    commute with the Z-type stabilizers.  Mathematically, this requirement can be written as

    H_x @ H_z.T == 0,

    where H_x and H_z are, respectively, the parity check matrices of the classical codes that
    define the X-type and Z-type stabilizers of the CSS code.  Note that H_x witnesses Z-type errors
    and H_z witnesses X-type errors.

    The full parity check matrix of a CSSCode is
    ⌈ H_z,  0  ⌉
    ⌊  0 , H_x ⌋.
    """

    code_x: ClassicalCode  # X-type parity checks, measuring Z-type errors
    code_z: ClassicalCode  # Z-type parity checks, measuring X-type errors

    _conjugate: slice | Sequence[int]
    _codes_equal: bool
    _logical_ops: galois.FieldArray | None = None
    _exact_distance_x: int | None = None
    _exact_distance_z: int | None = None

    def __init__(
        self,
        code_x: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_z: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] | None = (),
        skip_validation: bool = False,
    ) -> None:
        """Construct a CSS code from X-type and Z-type parity checks.

        Allow specifying local Fourier transformations on the qudits specified by `conjugate`.
        """
        self.code_x = ClassicalCode(code_x, field)
        self.code_z = ClassicalCode(code_z, field)
        if field is None and self.code_x.field is not self.code_z.field:
            raise ValueError("The sub-codes provided for this CSSCode are over different fields")
        self._field = self.code_x.field

        if not skip_validation and not self.is_valid:
            raise ValueError("The sub-codes provided for this CSSCode are incompatible")

        self._conjugated_qubits = conjugate or ()
        self._codes_equal = self.code_x == self.code_z

    @functools.cached_property
    def is_valid(self) -> bool:
        """Is this a valid CSS code?"""
        return self.code_x.num_bits == self.code_z.num_bits and not np.any(
            self.matrix_x @ self.matrix_z.T
        )

    @functools.cached_property
    def matrix(self) -> galois.FieldArray:
        """Overall parity check matrix."""
        matrix = np.block(
            [
                [self.matrix_z, np.zeros_like(self.matrix_z)],
                [np.zeros_like(self.matrix_x), self.matrix_x],
            ]
        )
        return self.field(self.conjugate(matrix, self.conjugated_qubits))

    @property
    def matrix_x(self) -> galois.FieldArray:
        """X-type parity checks."""
        return self.code_x.matrix

    @property
    def matrix_z(self) -> galois.FieldArray:
        """Z-type parity checks."""
        return self.code_z.matrix

    @property
    def conjugated_qubits(self) -> slice | Sequence[int]:
        """Which qubits are conjugated?"""
        return self._conjugated_qubits

    @property
    def num_checks_x(self) -> int:
        """Number of X-type parity checks in this code."""
        return self.matrix_x.shape[0]

    @property
    def num_checks_z(self) -> int:
        """Number of X-type parity checks in this code."""
        return self.matrix_z.shape[0]

    @property
    def num_checks(self) -> int:
        """Number of parity checks in this code."""
        return self.num_checks_x + self.num_checks_z

    @property
    def num_qudits(self) -> int:
        """Number of data qudits in this code."""
        return self.matrix_x.shape[1]

    @property
    def dimension(self) -> int:
        """Number of logical qudits encoded by this code."""
        return self.code_x.dimension + self.code_z.dimension - self.num_qudits

    def get_code_params(
        self, *, bound: int | bool | None = None, **decoder_args: object
    ) -> tuple[int, int, int, int]:
        """Compute the parameters of this code: [[n,k,d,w]].

        Here:
        - n is the number of data qudits
        - k is the number of encoded ("logical") qudits
        - d is the code distance
        - w is the maximal weight of (i.e., number of qudits addressed by) a parity check

        Keyword arguments are passed to the calculation of code distance.
        """
        distance = self.get_distance(pauli=None, bound=bound, vector=None, **decoder_args)
        return self.num_qudits, self.dimension, distance, self.get_weight()

    def get_distance(
        self,
        pauli: PauliXZ | None = None,
        *,
        bound: int | bool | None = None,
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: object,
    ) -> int:
        """Compute (or upper bound) the minimal weight of a nontrivial logical operator.

        If `bound is None`, compute an exact code distance by brute force.  Otherwise, compute an
        upper bound using a randomized algorithm described in arXiv:2308.07915, minimizing over
        `bound` random trials.  For a detailed explanation, see `CSSCode.get_one_distance_bound`.

        If provided a vector, compute the minimum Hamming distance between this vector and a
        (possibly trivial) X-type or Z-type logical operator, as applicable.

        Additional arguments, if applicable, are passed to a decoder in
        `CSSCode.get_one_distance_bound`.
        """
        if bound is None:
            return self.get_distance_exact(pauli, vector=vector)
        return self.get_distance_bound(pauli, num_trials=int(bound), vector=vector, **decoder_args)

    def get_distance_exact(
        self, pauli: PauliXZ | None, *, vector: Sequence[int] | npt.NDArray[np.int_] | None = None
    ) -> int:
        """Compute the minimal weight of a nontrivial code word by brute force.

        If provided a vector, compute the minimum Hamming distance between this vector and a
        (possibly trivial) X-type or Z-type logical operator, as applicable.
        """
        assert pauli is None or pauli in PAULIS_XZ
        if pauli is None:
            return min(
                self.get_distance_exact(Pauli.X, vector=vector),
                self.get_distance_exact(Pauli.Z, vector=vector),
            )

        if vector is not None:
            code_z = self.code_z if pauli == Pauli.X else self.code_x
            ops_x = code_z.words()
            vector = self.field(vector)
            return min(np.count_nonzero(word - vector) for word in ops_x)

        # if we know the exact code distance, return it
        if pauli == Pauli.X and self._exact_distance_x is not None:
            return self._exact_distance_x
        if pauli == Pauli.Z and self._exact_distance_z is not None:
            return self._exact_distance_z

        # we do not know the exact distance, so compute it
        code_x = self.code_x if pauli == Pauli.X else self.code_z
        code_z = self.code_z if pauli == Pauli.X else self.code_x
        dual_code_x = ~code_x
        nontrivial_ops_x = (word for word in code_z.words() if word not in dual_code_x)
        distance = min(np.count_nonzero(word) for word in nontrivial_ops_x)

        # save the exact distance and return
        if pauli == Pauli.X:
            self._exact_distance_x = distance
        if pauli == Pauli.Z:
            self._exact_distance_z = distance
        return distance

    def get_distance_bound(
        self,
        pauli: PauliXZ | None = None,
        num_trials: int = 1,
        *,
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: object,
    ) -> int:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        If provided a vector, compute the minimum Hamming distance between this vector and a
        (possibly trivial) X-type or Z-type logical operator, as applicable.

        Additional arguments, if applicable, are passed to a decoder in
        `CSSCode.get_one_distance_bound`.
        """
        distance_bounds = (
            self.get_one_distance_bound(pauli, vector=vector, **decoder_args)
            for _ in range(num_trials)
        )
        return min(distance_bounds, default=self.num_qudits)

    def get_one_distance_bound(
        self,
        pauli: PauliXZ | None = None,
        *,
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: object,
    ) -> int:
        """Compute a single upper bound on code distance.

        If provided a vector, compute the minimum Hamming distance between this vector and a
        (possibly trivial) X-type or Z-type logical operator, as applicable.

        Additional arguments, if applicable, are passed to a decoder.

        This method uses a randomized algorithm described in arXiv:2308.07915, and also below.

        For ease of language, we henceforth assume (without loss of generality) that we are
        computing an X-distance.

        Pick a random Z-type logical operator Z(w_z) whose support is indicated by the bistring w_z.
        We now wish to find a low-weight Pauli-X string X(w_x) that
            (a) has a trivial syndrome, and
            (b) anti-commutes with Z(w_z),
        which together would imply that X(w_x) is a nontrivial X-type logical operator.
        Mathematically, these conditions are equivalent to requiring that
            (a) H_z @ w_x = 0, and
            (b) w_z @ w_x = 1,
        where H_z is the parity check matrix of the Z-type subcode that witnesses X-type errors.

        Conditions (a) and (b) can be combined into the single block-matrix equation
            ⌈ H_z   ⌉         ⌈ 0 ⌉
            ⌊ w_z.T ⌋ @ w_x = ⌊ 1 ⌋,
        where the "0" on the top right is interpreted as a zero vector.  This equation can be solved
        by decoding the syndrome [ 0, 0, ..., 0, 1 ].T for the parity check matrix [ H_z.T, w_z ].T.

        We solve the above decoding problem with a decoder in `decode`.  If the decoder fails to
        find a solution, try again with a new random operator Z(w_z).  If the decoder succeeds in
        finding a solution w_x, this solution corresponds to a logical X-type operator X(w_x) -- and
        presumably one of low Hamming weight, since decoders try to find low-weight solutions.
        Return the Hamming weight |w_x|.
        """
        assert pauli is None or pauli in PAULIS_XZ
        pauli = pauli or random.choice(PAULIS_XZ)

        # define code_z and pauli_z as if we are computing X-distance
        code_z = self.code_z if pauli == Pauli.X else self.code_x
        pauli_z: Literal[Pauli.Z, Pauli.X] = Pauli.Z if pauli == Pauli.X else Pauli.X

        if vector is not None:
            # find the distance of the given vector from a logical X-type operator
            correction = qldpc.decoder.decode(
                code_z.matrix,
                code_z.matrix @ self.field(vector),
                **decoder_args,
            )
            return int(np.count_nonzero(correction))

        # construct the effective syndrome
        effective_syndrome = np.zeros(code_z.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1
        _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field, bound_index=-1)

        logical_op_found = False
        while not logical_op_found:
            # support of pauli string with a trivial syndrome
            word = self.get_random_logical_op(pauli_z, ensure_nontrivial=True)

            # support of a candidate pauli-type logical operator
            effective_check_matrix = np.vstack([code_z.matrix, word]).view(np.ndarray)
            candidate_logical_op = qldpc.decoder.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )

            # check whether decoding was successful
            actual_syndrome = effective_check_matrix @ candidate_logical_op % self.field.order
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        # return the Hamming weight of the logical operator
        return int(np.count_nonzero(candidate_logical_op))

    def get_logical_ops(self, pauli: PauliXZ | None = None) -> galois.FieldArray:
        """Complete basis of nontrivial X-type and Z-type logical operators for this code.

        Logical operators are represented by a three-dimensional array `logical_ops` with dimensions
        (2, k, n), where k and n are respectively the numbers of logical and physical qudits in this
        code.  The bitstring `logical_ops[0, 4, :]`, for example, indicates the support (i.e., the
        physical qudits addressed nontrivially) by the logical Pauli-X operator on logical qudit 4.

        If passed a pauli operator (Pauli.X or Pauli.Z), return the two-dimensional array of logical
        operators of the specified type.

        In the case of qudits with dimension > 2, the "Pauli-X" and "Pauli-Z" operators constructed
        by this method are the unit shift and phase operators that generate all logical X-type and
        Z-type qudit operators.

        Logical operators are constructed using the method described in Section 4.1 of Gottesman's
        thesis (arXiv:9705052), slightly modified and generalized for qudits.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve logical operators of one type only
        if pauli is not None:
            return self.get_logical_ops()[pauli]

        # memoize manually because other methods may modify the logical operators computed here
        if self._logical_ops is not None:
            return self._logical_ops

        num_qudits = self.num_qudits
        dimension = self.dimension
        identity = self.field.Identity(dimension)

        def row_reduce(
            matrix: npt.NDArray[np.int_],
        ) -> tuple[npt.NDArray[np.int_], Sequence[int], Sequence[int]]:
            """Perform Gaussian elimination on the matrix.

            Returns:
                matrix_RRE: the reduced row echelon form of the matrix.
                pivot: the "pivot" columns of the reduced matrix.
                other: the remaining columns of the reduced matrix.

            In reduced row echelon form, the first nonzero entry of each row is a 1, and these 1s
            occur at a unique columns for each row; these columns are the "pivots" of matrix_RRE.
            """
            # row-reduce the matrix and identify its pivots
            matrix_RRE = self.field(matrix).row_reduce()
            pivots = (matrix_RRE != 0).argmax(axis=1)

            # remove trailing zero pivots, which correspond to trivial (all-zero) rows
            if pivots.size > 1 and pivots[-1] == 0:
                pivots = np.concatenate([[pivots[0]], pivots[1:][pivots[1:] != 0]])

            # identify remaining columns and return
            other = [qq for qq in range(matrix.shape[1]) if qq not in pivots]
            return matrix_RRE, pivots, other

        # identify check matrices for X/Z-type errors, and the current qudit locations
        checks_x: npt.NDArray[np.int_] = self.matrix_z
        checks_z: npt.NDArray[np.int_] = self.matrix_x
        qudit_locs = np.arange(num_qudits, dtype=int)

        # row reduce the check matrix for X-type errors and move its pivots to the back
        checks_x, pivot_x, other_x = row_reduce(checks_x)
        checks_x = np.hstack([checks_x[:, other_x], checks_x[:, pivot_x]])
        checks_z = np.hstack([checks_z[:, other_x], checks_z[:, pivot_x]])
        qudit_locs = np.hstack([qudit_locs[other_x], qudit_locs[pivot_x]])

        # row reduce the check matrix for Z-type errors and move its pivots to the back
        checks_z, pivot_z, other_z = row_reduce(checks_z)
        checks_x = np.hstack([checks_x[:, other_z], checks_x[:, pivot_z]])
        checks_z = np.hstack([checks_z[:, other_z], checks_z[:, pivot_z]])
        qudit_locs = np.hstack([qudit_locs[other_z], qudit_locs[pivot_z]])

        # run some sanity checks
        assert pivot_z[-1] < num_qudits - len(pivot_x)
        assert dimension + len(pivot_x) + len(pivot_z) == num_qudits

        # get the support of the check matrices on non-pivot qudits
        non_pivot_x = checks_x[: len(pivot_x), :dimension]
        non_pivot_z = checks_z[: len(pivot_z), :dimension]

        # construct logical X operators
        logicals_x = self.field.Zeros((dimension, num_qudits))
        logicals_x[:, dimension : dimension + len(pivot_x)] = -non_pivot_x.T
        logicals_x[:dimension, :dimension] = identity

        # construct logical Z operators
        logicals_z = self.field.Zeros((dimension, num_qudits))
        logicals_z[:, -len(pivot_z) :] = -non_pivot_z.T
        logicals_z[:dimension, :dimension] = identity

        # move qudits back to their original locations
        permutation = np.argsort(qudit_locs)
        logicals_x = logicals_x[:, permutation]
        logicals_z = logicals_z[:, permutation]

        self._logical_ops = self.field(np.stack([logicals_x, logicals_z]))
        return self._logical_ops

    def get_random_logical_op(
        self, pauli: PauliXZ, *, ensure_nontrivial: bool = False
    ) -> galois.FieldArray:
        """Return a random logical operator of a given type.

        A random logical operator may be trivial, which is to say that it may be equal to the
        identity modulo stabilizers.  If `ensure_nontrivial is True`, ensure that the logical
        operator we return is nontrivial.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        if not ensure_nontrivial:
            return (self.code_z if pauli == Pauli.X else self.code_x).get_random_word()

        # generate random logical ops until we find ones with a nontrivial commutation relation
        noncommuting_ops_found = False
        while not noncommuting_ops_found:
            op_a = self.get_random_logical_op(pauli, ensure_nontrivial=False)
            op_b = self.get_random_logical_op(
                ~pauli, ensure_nontrivial=False  # type:ignore[arg-type]
            )
            noncommuting_ops_found = bool(np.any(op_a @ op_b))

        return op_a

    def reduce_logical_op(self, pauli: PauliXZ, logical_index: int, **decoder_args: object) -> None:
        """Reduce the weight of a logical operator.

        A minimal-weight logical operator is found by enforcing that it has a trivial syndrome, and
        that it commutes with all logical operators except its dual.  This is essentially the same
        method as that used in CSSCode.get_one_distance_bound.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        assert 0 <= logical_index < self.dimension

        # effective check matrix = syndromes and other logical operators
        code = self.code_z if pauli == Pauli.X else self.code_x
        all_dual_ops = self.get_logical_ops(~pauli)  # type:ignore[arg-type]
        effective_check_matrix = np.vstack([code.matrix, all_dual_ops]).view(np.ndarray)
        dual_op_index = code.num_checks + logical_index

        # enforce that the new logical operator commutes with everything except its dual
        effective_syndrome = np.zeros((code.num_checks + self.dimension), dtype=int)
        effective_syndrome[dual_op_index] = 1
        _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field, bound_index=dual_op_index)

        logical_op_found = False
        while not logical_op_found:
            candidate_logical_op = qldpc.decoder.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )
            actual_syndrome = effective_check_matrix @ candidate_logical_op % self.field.order
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        assert self._logical_ops is not None
        self._logical_ops[pauli, logical_index] = candidate_logical_op

    def reduce_logical_ops(self, pauli: PauliXZ | None = None, **decoder_args: object) -> None:
        """Reduce the weight of all logical operators."""
        assert pauli is None or pauli in PAULIS_XZ
        if pauli is None:
            self.reduce_logical_ops(Pauli.X, **decoder_args)
            self.reduce_logical_ops(Pauli.Z, **decoder_args)
        else:
            for logical_index in range(self.dimension):
                self.reduce_logical_op(pauli, logical_index, **decoder_args)


def _fix_decoder_args_for_nonbinary_fields(
    decoder_args: dict[str, object], field: type[galois.FieldArray], bound_index: int | None = None
) -> None:
    """Fix decoder arguments for nonbinary number fields.

    If the field has order greater than 2, then we can only decode
    (a) prime number fields, with
    (b) an integer-linear program decoder.

    If provided a bound_index, treat the constraint corresponding to this row of the parity check
    matrix as a lower bound (>=) rather than a strict equality (==) constraint.
    """
    if field.order > 2:
        if field.degree > 1:
            raise ValueError("Method only supported for prime number fields")
        decoder_args["with_ILP"] = True
        decoder_args["modulus"] = field.order
        if bound_index is not None:
            decoder_args["lower_bound_row"] = bound_index


################################################################################
# bicycle and quasi-cyclic codes


class GBCode(CSSCode):
    """Generalized bicycle (GB) code.

    A GBCode code is built out of two matrices A and B, which are combined as
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T],
    to form the parity check matrices of a CSSCode.  If A and B commute, the parity check matrices
    matrix_x and matrix_z satisfy the requirements of a CSSCode.

    References:
    - https://arxiv.org/abs/2012.04068
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        matrix_b: npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] = (),
    ) -> None:
        """Construct a generalized bicycle code."""
        if matrix_b is None:
            matrix_b = matrix_a  # pragma: no cover

        code_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        matrix_a = code_field(matrix_a)
        matrix_b = code_field(matrix_b)
        if not np.array_equal(matrix_a @ matrix_b, matrix_b @ matrix_a):
            raise ValueError("The matrices provided for this GBCode are incompatible")

        matrix_x = np.block([matrix_a, matrix_b])
        matrix_z = np.block([matrix_b.T, -matrix_a.T])
        CSSCode.__init__(self, matrix_x, matrix_z, field, conjugate=conjugate, skip_validation=True)


QuasiCyclicPlaquetteMap = Callable[[int, int, int | PauliXZ], tuple[int, int]]


class QCCode(GBCode):
    """Quasi-cyclic (QC) codes from arXiv:2308.07915.

    A quasi-cyclic code is a CSS code with subcode parity check matrices
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T],
    where A = A_{ij} x^i y^j and B = B_{ij} x^i y^j are bivariate polynomials.  Here:
    - A_{ij} and B_{ij} are scalar coefficients (over some finite field),
    - x generates a group of order R_x, and
    - y generates a group of order R_y.

    A quasi-cyclic code is defined by...
    [1] two cyclic group orders, and
    [2] two sympy polynomials in two variables.
    By default, group orders are associated in lexicographic order with free variables of the
    polynomials.  Group orders can also be assigned to variables explicitly with a dictionary.
    """

    def __init__(
        self,
        orders: tuple[int, int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic | None = None,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Construct a quasi-cyclic code."""
        if poly_b is None:
            poly_b = poly_a  # pragma: no cover
        self.poly_a = sympy.Poly(poly_a)
        self.poly_b = sympy.Poly(poly_b)

        # identify the symbols used to denote cyclic group generators
        symbols = poly_a.free_symbols | poly_b.free_symbols
        if len(symbols) < len(orders) or (
            isinstance(orders, dict) and any(symbol not in orders for symbol in symbols)
        ):
            raise ValueError(f"Could not match symbols {symbols} to group orders {orders}")

        # identify cyclic group orders with symbols in the polynomials
        if not isinstance(orders, dict):
            orders_dict = {}
            for symbol, order in zip(sorted(symbols, key=str), orders):
                assert isinstance(symbol, sympy.Symbol), f"Invalid symbol: {symbol}"
                orders_dict[symbol] = order
            orders = orders_dict
        self.symbols = tuple(orders.keys())
        self.orders = tuple(orders.values())

        # identify the group generator associated with each symbol
        self.group = abstract.AbelianGroup(*orders.values(), product_lift=True)
        self.gens = self.group.generators
        self.symbol_gens = dict(zip(self.symbols, self.gens))

        # hadamard-transform qubits in the "R" sector
        num_qudits = self.group.order * 2
        qudits_to_conjugate: slice | Sequence[int] = (
            slice(num_qudits // 2, num_qudits + 1) if conjugate else ()
        )

        # build defining matrices of a generalized bicycle code
        matrix_a = self.eval(self.poly_a).lift().view(np.ndarray)
        matrix_b = self.eval(self.poly_b).lift().view(np.ndarray)
        GBCode.__init__(self, matrix_a, matrix_b, field, conjugate=qudits_to_conjugate)

    def eval(
        self,
        expr: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul | sympy.Poly,
    ) -> abstract.Element:
        """Convert a sympy expression into an element of a group algebra."""
        # evaluate simple cases
        if isinstance(expr, sympy.Integer):
            return int(expr) * abstract.Element(self.group, self.to_group_member(expr))
        if isinstance(expr, (sympy.Symbol, sympy.Pow)):
            return abstract.Element(self.group, self.to_group_member(expr))

        # evaluate a product or polynomial
        element = abstract.Element(self.group)
        for term in expr.as_expr().args:
            element += functools.reduce(
                abstract.Element.__mul__,
                [self.eval(factor) for factor in term.as_ordered_factors()],
            )
        return element

    def to_group_member(
        self, expr: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul
    ) -> abstract.GroupMember:
        """Convert a sympy expression into an associated member of this code's base group."""
        if isinstance(expr, sympy.Integer):
            return self.group.identity
        if isinstance(expr, sympy.Symbol):
            return self.symbol_gens[expr]
        if isinstance(expr, sympy.Pow):
            base, exp = expr.as_base_exp()
            return self.symbol_gens[base] ** exp
        if isinstance(expr, sympy.Mul):
            output = self.group.identity
            for factor in expr.args:
                if not isinstance(factor, sympy.Integer):
                    base, exp = factor.as_base_exp()
                    output *= self.symbol_gens[base] ** exp
            return output
        return NotImplemented  # pragma: no cover

    def get_exponents(
        self, expr: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul
    ) -> tuple[int, int]:
        """Get the exponents of a term, for example converting x**2 y**4 into (2, 4)."""
        exponents = {}
        if isinstance(expr, sympy.Symbol):
            exponents[expr] = 1
        elif isinstance(expr, sympy.Pow):
            base, exp = expr.as_base_exp()
            exponents[base] = exp
        elif isinstance(expr, sympy.Mul):
            for factor in expr.args:
                base, exp = factor.as_base_exp()
                exponents[base] = exp
        return exponents.get(self.symbols[0], 0), exponents.get(self.symbols[1], 0)

    @functools.cache
    def get_toric_mappings(self) -> Sequence[tuple[QuasiCyclicPlaquetteMap, tuple[int, int]]]:
        """Get plaquette mappings that arrange qubits in a toric layout.

        Each plaquette looks like:
            L X
            Z R
        where L and R are data qubits, and X and Z are checks.  In a toric layout, plaquettes are
        arranged in a grid, and each check addresses all of its neighboring data qubits, as well as
        a few far-away qubits.  This method returns a "plaquette mapping" and torus dimensions
        (shape) for all toric layouts of this code.

        Each plaquette on the torus is nominally indexed by coordinates (i, j).  Each plaquette
        mapping then takes:
        - the plaquettes x-coordinate i
        - the plaquettes y-coordinate j
        - a qubit sector: 0, 1, Pauli.X, or Pauli.Z (respectively, for L, R, X, or Z),
        - the dimensions (shape) of the toric layout (a pair of integers for torus width/height).
        The plaquette mapping returns the coordinates of the new plaquette for the specified qubit.
        """
        if not nx.is_weakly_connected(self.graph):
            # a connected tanner graph is a baseline requirement for a toric mapping to exist
            return []

        # identify individual terms in the polynomials
        terms_a = self.poly_a.as_expr().args
        terms_b = self.poly_b.as_expr().args

        # find combinations of terms that enable a toric layout
        toric_params = []
        for (a_1, a_2), (b_1, b_2) in itertools.product(
            itertools.combinations(terms_a, 2), itertools.combinations(terms_b, 2)
        ):
            gen_a = self.to_group_member(a_1 * a_2 ** (-1))
            gen_b = self.to_group_member(b_1 * b_2 ** (-1))
            if (
                gen_a.order() * gen_b.order() == self.group.order
                and comb.PermutationGroup(gen_a, gen_b).order() == self.group.order
            ):
                toric_params.append((a_1, a_2, b_1, b_2))

        # identify torus shapes and qubit-to-plaquette mappings
        layout_data = []
        for a_1, a_2, b_1, b_2 in toric_params:
            shift_a = a_1 * a_2 ** (-1)
            shift_b = b_1 * b_2 ** (-1)
            """
            For generators of the form
                g = x^p y^q  <-- shift_a,
                h = x^u y^v  <-- shift_b,
            build a grid_map (dictionary) that maps (i, j) --> (a, b), where
                x^i y^j = g^a h^b.
            Equivalently, we want
                i = a p + b u  mod order(x),
                j = b q + b v  mod order(y).
            """
            gen_g = self.to_group_member(shift_a)
            gen_h = self.to_group_member(shift_b)
            pp, qq = self.get_exponents(shift_a)
            uu, vv = self.get_exponents(shift_b)
            torus_shape: tuple[int, int] = (int(gen_g.order()), int(gen_h.order()))
            grid_map = {
                (
                    (aa * pp + bb * uu) % self.orders[0],
                    (aa * qq + bb * vv) % self.orders[1],
                ): (aa, bb)
                for aa, bb in np.ndindex(torus_shape)
            }
            # figure out how to shift qubits in each sector:
            # (0 <--> L) or (1 <--> R) for data qubits, and X or Z for check qubits
            sector_shifts = {
                0: (0, 0),  # "L" data qubits
                1: self.get_exponents(a_2 ** (-1) * b_1),  # "R" data qubits
                Pauli.X: self.get_exponents(a_2 ** (-1)),  # "X" check qubits
                Pauli.Z: self.get_exponents(b_1),  # "Z" check qubits
            }

            plaquette_map = functools.partial(
                self.full_plaquette_map,
                grid_map=grid_map,
                sector_shifts=sector_shifts,
                torus_shape=torus_shape,
            )
            layout_data.append((plaquette_map, torus_shape))

        return layout_data

    def full_plaquette_map(
        self,
        ii: int,
        jj: int,
        qubit_sector: int | PauliXZ,
        grid_map: dict[tuple[int, int], tuple[int, int]],
        sector_shifts: dict[int | PauliXZ, tuple[int, int]],
        torus_shape: tuple[int, int],
    ) -> tuple[int, int]:
        """Map from "original" plaquette coordinates to "shifted" plaquette coordinates."""
        s_i = (ii - sector_shifts[qubit_sector][0]) % self.orders[0]
        s_j = (jj - sector_shifts[qubit_sector][1]) % self.orders[1]
        s_a, s_b = grid_map[s_i, s_j]
        return s_a % torus_shape[0], s_b % torus_shape[1]

    def get_toric_checks(
        self, plaquette_map: QuasiCyclicPlaquetteMap, torus_shape: tuple[int, int]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-type and Z-type parity check matrices for a toric layout."""

        # loop over each of X-type and Z-type parity checks
        for pauli in PAULIS_XZ:
            matrix = self.matrix_x if pauli == Pauli.X else self.matrix_z
            old_checks = matrix.reshape(*self.orders, 2, *self.orders)
            new_checks = self.field.Zeros((*torus_shape, 2, *torus_shape))

            # loop over every check
            for c_i, c_j in np.ndindex(*self.orders):
                c_a, c_b = plaquette_map(c_i, c_j, pauli)

                # loop over every qubit
                for sector, d_i, d_j in zip(*np.where(old_checks[c_i, c_j])):
                    d_a, d_b = plaquette_map(d_i, d_j, sector)
                    new_checks[c_a, c_b, sector, d_a, d_b] = old_checks[c_i, c_j, sector, d_i, d_j]

            # save the shifted checks to an X/Z parity check matrix as appropriate
            if pauli == Pauli.X:
                matrix_x = new_checks.reshape(self.matrix_x.shape)
            else:
                assert pauli == Pauli.Z
                matrix_z = new_checks.reshape(self.matrix_z.shape)

        return matrix_x, matrix_z

    @classmethod
    def get_toric_qubit_pos(
        cls,
        aa: int,
        bb: int,
        sector: int | PauliXZ,
        torus_shape: tuple[int, int] | None = None,
        open_boundaries: bool = False,
    ) -> tuple[int, int]:
        """Get the position of a qubit in the given sector of plaquette (aa, bb) on a torus.

        If open_boundaries=True, "fold" the torus for a qubit layout with open boundaries.
        """
        if torus_shape is not None:
            aa = aa % torus_shape[0]
            bb = bb % torus_shape[1]
        xx = 2 * aa + int(sector in [Pauli.X, 1])
        yy = 2 * bb + int(sector in [Pauli.Z, 1])
        if open_boundaries:
            assert torus_shape is not None, "Cannot fold a torus without knowing its shape"
            xx = 2 * xx if xx < torus_shape[0] else (2 * torus_shape[0] - 1 - xx) * 2 + 1
            yy = 2 * yy if yy < torus_shape[1] else (2 * torus_shape[1] - 1 - yy) * 2 + 1
        return int(xx), int(yy)

    def get_check_shifts(
        self,
        plaquette_map: QuasiCyclicPlaquetteMap,
        torus_shape: tuple[int, int],
        open_boundaries: bool = False,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Get the relative positions of data qubits addressed by X-type and Z-type check qubits.

        If open_boundaries=True, "fold" the torus for a qubit layout with open boundaries.
        """
        # identify the parity check matrices
        matrix_x, matrix_z = self.get_toric_checks(plaquette_map, torus_shape)

        # Identify the plaquettes on which we need to examine check qubits.  If we have periodic
        # boundaries, all plaquettes "look the same", so we only need to consider one of them.
        # Otherwise, we generally need to consider all plaquettes.
        plaquettes = [(0, 0)] if not open_boundaries else [*np.ndindex(*torus_shape)]

        # sets of relative coordinates, organized by stabilizer type
        shifts: dict[PauliXZ, set[tuple[int, int]]] = {}
        for pauli in PAULIS_XZ:
            shifts[pauli] = set()

            # organize checks by plaquette on the torus
            shape = (*torus_shape, 2, *torus_shape)
            checks = (matrix_x if pauli == Pauli.X else matrix_z).reshape(shape)

            # loop over all plaquettes we need to consider
            for p_a, p_b in plaquettes:

                # identify the location of a check qubit, and the support of its stabilizer
                c_a, c_b = self.get_toric_qubit_pos(p_a, p_b, pauli, torus_shape, open_boundaries)
                check = checks[p_a, p_b]

                # identify the relative position of all data qubits addressed by this check
                for sector, aa, bb in zip(*np.where(check)):
                    # position of this data qubit
                    d_a, d_b = self.get_toric_qubit_pos(
                        aa, bb, sector, torus_shape, open_boundaries
                    )

                    # relative position of data qubit from check qubitF
                    shift_a = (d_a - c_a) % (2 * torus_shape[0])
                    shift_b = (d_b - c_b) % (2 * torus_shape[1])
                    shift_a = shift_a if shift_a <= torus_shape[0] else shift_a - 2 * torus_shape[0]
                    shift_b = shift_b if shift_b <= torus_shape[1] else shift_b - 2 * torus_shape[1]
                    shifts[pauli].add((shift_a, shift_b))

        return shifts[Pauli.X], shifts[Pauli.Z]


################################################################################
# hypergraph and lifted product codes


class HGPCode(CSSCode):
    """Hypergraph product (HGP) code.

    A hypergraph product code AB is constructed from two classical codes, A and B.

    Consider the following:
    - Code A has 3 data and 2 check bits.
    - Code B has 4 data and 3 check bits.
    We represent data bits/qudits by circles (○) and check bits/qudits by squares (□).

    Denode the Tanner graph of code C by G_C.  The nodes of G_AB can be arranged into a matrix.  The
    rows of this matrix are labeled by nodes of G_A, and columns by nodes of G_B.  The matrix of
    nodes in G_AB can thus be organized into four sectors:

    ――――――――――――――――――――――――――――――――――
      | ○ ○ ○ ○ | □ □ □ ← nodes of G_B
    ――+―――――――――+――――――
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ――+―――――――――+――――――
    □ | □ □ □ □ | ○ ○ ○
    □ | □ □ □ □ | ○ ○ ○
    ↑ nodes of G_A
    ――――――――――――――――――――――――――――――――――

    We identify each sector by two bits.
    In the example above:
    - sector (0, 0) has 3×4=12 data qudits
    - sector (0, 1) has 3×3=9 check qudits
    - sector (1, 0) has 2×4=8 check qudits
    - sector (1, 1) has 2×3=6 data qudits

    Edges in G_AB are inherited across rows/columns from G_A and G_B.  For example, if rows r_1 and
    r_2 share an edge in G_A, then the same is true in every column of G_AB.

    By default, the check qudits in sectors (0, 1) of G_AB measure Z-type operators.  Likewise with
    sector (1, 0) and X-type operators.  If a HGP is constructed with `conjugate==True`, then the
    types of operators addressing the nodes in sector (1, 1) are switched.

    This class contains two equivalent constructions of an HGPCode:
    - A construction based on Tanner graphs (as discussed above).
    - A construction based on check matrices, taken from arXiv:2202.01702.
    The latter construction is less intuitive, but more efficient.

    References:
    - https://errorcorrectionzoo.org/c/hypergraph_product
    - https://arxiv.org/abs/2202.01702
    - https://www.youtube.com/watch?v=iehMcUr2saM
    - https://arxiv.org/abs/0903.0566
    - https://arxiv.org/abs/1202.0928
    """

    sector_size: npt.NDArray[np.int_]

    def __init__(
        self,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Hypergraph product of two classical codes, as in arXiv:2202.01702.

        The parity check matrices of the hypergraph product code are:

        matrix_x = [ H1 ⨂ In2, Im1 ⨂ H2.T]
        matrix_z = [-In1 ⨂ H2, H1.T ⨂ Im2]

        Here (H1, H2) == (matrix_a, matrix_b), and I[m/n][1/2] are identity matrices,
        with (m1, n1) = H1.shape and (m2, n2) = H2.shape.

        A minus sign in one sector of matrix_x or matrix_z is necessary to satisfy CSS code
        requirements with nonbinary fields.  The placement of this sign is chosen for consistency
        with the tensor product of chain complexes.
        """
        if code_b is None:
            code_b = code_a
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b, field)
        field = code_a.field.order

        # use a matrix-based hypergraph product to identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(code_a.matrix, code_b.matrix)

        # identify the number of qudits in each sector
        self.sector_size = np.outer(
            [code_a.num_bits, code_a.num_checks],
            [code_b.num_bits, code_b.num_checks],
        )

        # identify which qudits to conjugate (Hadamard-transform)
        qudits_to_conjugate = slice(self.sector_size[0, 0], None) if conjugate else None

        CSSCode.__init__(
            self,
            matrix_x.astype(int),
            matrix_z.astype(int),
            field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )

    @classmethod
    def get_matrix_product(
        cls,
        matrix_a: npt.NDArray[np.int_ | np.object_],
        matrix_b: npt.NDArray[np.int_ | np.object_],
    ) -> tuple[npt.NDArray[np.int_ | np.object_], npt.NDArray[np.int_ | np.object_]]:
        """Hypergraph product of two parity check matrices."""
        # construct the nontrivial blocks of the final parity check matrices
        mat_H1_In2 = np.kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        mat_In1_H2 = np.kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        mat_H1_Im2_T = np.kron(matrix_a.T, np.eye(matrix_b.shape[0], dtype=int))
        mat_Im1_H2_T = np.kron(np.eye(matrix_a.shape[0], dtype=int), matrix_b.T)

        # construct the X-sector and Z-sector parity check matrices
        matrix_x = np.block([mat_H1_In2, mat_Im1_H2_T])
        matrix_z = np.block([-mat_In1_H2, mat_H1_Im2_T])
        return matrix_x, matrix_z

    @classmethod
    def get_graph_product(
        cls, graph_a: nx.DiGraph, graph_b: nx.DiGraph, *, conjugate: bool = False
    ) -> nx.DiGraph:
        """Hypergraph product of two Tanner graphs."""

        # start with a cartesian products of the input graphs
        graph_product = nx.cartesian_product(graph_a, graph_b)

        # fix edge orientation, and tag each edge with a QuditOperator
        graph = nx.DiGraph()
        for node_fst, node_snd, data in graph_product.edges(data=True):
            # determine which node is a check node vs. a qudit node
            if node_fst[0].is_data == node_fst[1].is_data:
                # the first node is in the (0, 0) or (1, 1) sector --> a data node
                node_qudit, node_check = node_fst, node_snd
            else:
                # the first node is in the (0, 1) or (1, 0) sector --> a check node
                node_check, node_qudit = node_fst, node_snd
            graph.add_edge(node_check, node_qudit)

            # by default, this edge is Z-type iff the check qudit is in the (0, 1) sector
            op = QuditOperator((0, data.get("val", 1)))
            if node_check[0].is_data:
                # make this a X-type operator
                op = ~op

            # for a conjugated code, flip X <--> Z operators in the (1, 1) sector
            if conjugate and not node_qudit[0].is_data:
                op = ~op

            # account for the minus sign in the (0, 0) sector of the Z-type subcode
            if node_qudit[0].is_data and node_check[0].is_data:
                op = -op

            graph[node_check][node_qudit][QuditOperator] = op

        # relabel nodes, from (node_a, node_b) --> node_combined
        node_map = HGPCode.get_product_node_map(graph_a.nodes, graph_b.nodes)
        graph = nx.relabel_nodes(graph, node_map)

        # remember order of the field, and use Pauli operators if appropriate
        if hasattr(graph_a, "order"):
            graph.order = graph_a.order
            if graph.order == 2:
                for _, __, data in graph.edges(data=True):
                    data[Pauli] = Pauli(data[QuditOperator].value)
                    del data[QuditOperator]

        return graph

    @classmethod
    def get_product_node_map(
        cls, nodes_a: Collection[Node], nodes_b: Collection[Node]
    ) -> dict[tuple[Node, Node], Node]:
        """Map (dictionary) that re-labels nodes in the hypergraph product of two codes."""
        index_qudit = 0
        index_check = 0
        node_map = {}
        for node_a, node_b in itertools.product(sorted(nodes_a), sorted(nodes_b)):
            if node_a.is_data == node_b.is_data:
                # this is a data qudit in sector (0, 0) or (1, 1)
                node = Node(index=index_qudit, is_data=True)
                index_qudit += 1
            else:
                # this is a check qudit in sector (0, 1) or (1, 0)
                node = Node(index=index_check, is_data=False)
                index_check += 1
            node_map[node_a, node_b] = node
        return node_map


class LPCode(CSSCode):
    """Lifted product (LP) code.

    A lifted product code is essentially the same as a hypergraph product code, except that the
    parity check matrices are "protographs", or matrices whose entries are members of a group
    algebra over a finite field F_q.  Each of these entries can be "lifted" to a representation as
    orthogonal matrices over F_q, in which case the protograph is interpreted as a block matrix;
    this is called "lifting" the protograph.

    Notes:
    - A lifted product code with protographs of size 1×1 is a generalized bicycle code.
    - A lifted product code with protographs whose entries get lifted to 1×1 matrices is a
        hypergraph product code of the lifted protographs.
    - One way to get an LPCode: take a classical code with parity check matrix H and multiply it by
        a diagonal matrix D = diag(a_1, a_2, ... a_n), where all {a_j} are elements of a group
        algebra.  The protograph P = H @ D can then be used for one of the protographs of an LPCode.

    References:
    - https://errorcorrectionzoo.org/c/lifted_product
    - https://arxiv.org/abs/2202.01702
    - https://arxiv.org/abs/2012.04068
    """

    def __init__(
        self,
        protograph_a: npt.NDArray[np.object_] | Sequence[Sequence[object]],
        protograph_b: npt.NDArray[np.object_] | Sequence[Sequence[object]] | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Lifted product of two protographs, as in arXiv:2012.04068."""
        if protograph_b is None:
            protograph_b = protograph_a
        protograph_a = abstract.Protograph(protograph_a)
        protograph_b = abstract.Protograph(protograph_b)
        field = protograph_a.field.order

        # identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(protograph_a, protograph_b)

        # identify the number of qudits in each sector
        self.sector_size = protograph_a.group.lift_dim * np.outer(
            protograph_a.shape[::-1],
            protograph_b.shape[::-1],
        )

        # identify which qudits to conjugate (Hadamard-transform)
        qudits_to_conjugate = slice(self.sector_size[0, 0], None) if conjugate else None

        CSSCode.__init__(
            self,
            abstract.Protograph(matrix_x.astype(object)).lift(),
            abstract.Protograph(matrix_z.astype(object)).lift(),
            field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )


################################################################################
# classical and quantum Tanner codes


class TannerCode(ClassicalCode):
    """Classical Tanner code, as described in DOI:10.1109/TIT.1981.1056404.

    A Tanner code T(G,C) is constructed from:
    [1] A bipartite "half-regular" graph G.  That is, a graph...
        ... with two sets of nodes, V and W.
        ... in which all nodes in V have degree n.
    [2] A classical code C on n bits.

    For convenience, we make G directed, with edges directed from V to W.  The node sets V and W can
    then be identified, respectively, by the sources and sinks of G.

    The Tanner code T(G,C) is defined on |W| bits.  A |W|-bit string x is a code word of T(G,C) iff,
    for every node v in V, the bits of x incident to v are a code word of C.

    This construction requires an ordering the edges E(v) adjacent to each vertex v.  This class
    sorts E(v) by the value of the "sort" attribute attached to each edge.  If there is no "sort"
    attribute, its value is treated as corresponding neighbor of v.

    Tanner codes can similarly be defined on regular (undirected) graphs G' = (V',E') by placing
    checks on V' and bits on E'.

    Notes:
    - If the subcode C has m checks, its parity matrix has shape (m,n).
    - The code T(G,C) has |W| bits and |V|m checks.
    """

    subgraph: nx.DiGraph
    subcode: ClassicalCode

    def __init__(self, subgraph: nx.Graph, subcode: ClassicalCode) -> None:
        """Construct a classical Tanner code."""
        if not isinstance(subgraph, nx.DiGraph):
            subgraph = TannerCode.as_directed_subgraph(subgraph)

        self.subgraph = subgraph
        self.subcode = subcode
        sources = [node for node in subgraph if subgraph.in_degree(node) == 0]
        sinks = [node for node in subgraph if subgraph.out_degree(node) == 0]
        sink_indices = {sink: idx for idx, sink in enumerate(sorted(sinks))}

        num_bits = len(sinks)
        num_checks = len(sources) * subcode.num_checks
        matrix = np.zeros((num_checks, num_bits), dtype=int)
        for idx, source in enumerate(sorted(sources)):
            checks = range(subcode.num_checks * idx, subcode.num_checks * (idx + 1))
            bits = [sink_indices[sink] for sink in self._get_sorted_neighbors(source)]
            matrix[np.ix_(checks, bits)] = subcode.matrix
        ClassicalCode.__init__(self, matrix, subcode.field.order)

    def _get_sorted_neighbors(self, node: object) -> Sequence[object]:
        """Sorted neighbors of the given node."""
        return sorted(
            self.subgraph.neighbors(node),
            key=lambda neighbor: self.subgraph[node][neighbor].get("sort", neighbor),
        )

    @classmethod
    def as_directed_subgraph(self, subgraph: nx.Graph) -> nx.DiGraph:
        """Convert an undirected graph for a Tanner code into a directed graph for the same code."""
        directed_subgraph = nx.DiGraph()
        for node_a, node_b, edge_data in subgraph.edges(data=True):
            edge = frozenset([node_a, node_b])
            directed_subgraph.add_edge(node_a, edge)
            directed_subgraph.add_edge(node_b, edge)
            if (sort_data := edge_data.pop("sort", None)) is not None:
                directed_subgraph[node_a][edge]["sort"] = sort_data[node_a]
                directed_subgraph[node_b][edge]["sort"] = sort_data[node_b]
        return directed_subgraph


# TODO: investigate construction in
# https://github.com/errorcorrectionzoo/eczoo_data/files/9210173/rotated.pdf
# see also Section 7 of https://arxiv.org/abs/2206.07571
class QTCode(CSSCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex.

    Altogether, a quantum Tanner code is defined by:
    - two symmetric (self-inverse) subsets A and B of a group G, and
    - two classical codes C_A and C_B, respectively with block lengths |A| and |B|.

    The X-type parity checks of a quantum Tanner code are the checks of a classical Tanner code
    whose generating graph is the subgraph_x of the Cayley complex (A,B).  The subcode of this
    classical Tanner code is ~(C_A ⨂ C_B), where ~C is the dual code to C.

    The Z-type parity checks are similarly defined with subgraph_z and subcode ~(~C_A ⨂ ~C_B).

    Notes:
    - "Good" quantum Tanner code: projective special linear group and random classical codes.

    References:
    - https://errorcorrectionzoo.org/c/quantum_tanner
    - https://arxiv.org/abs/2206.07571
    - https://arxiv.org/abs/2202.13641
    """

    complex: CayleyComplex

    def __init__(
        self,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember],
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        bipartite: bool = False,
        conjugate: slice | Sequence[int] | None = (),
    ) -> None:
        """Construct a quantum Tanner code."""
        if code_b is None:
            code_b = code_a
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b, field)
        if field is None and code_a.field is not code_b.field:
            raise ValueError("The sub-codes provided for this QTCode are over different fields")

        self.complex = CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        assert code_a.num_bits == len(self.complex.subset_a)
        assert code_b.num_bits == len(self.complex.subset_b)

        subgraph_x, subgraph_z = QTCode.get_subgraphs(self.complex)
        subcode_x = ~ClassicalCode.tensor_product(code_a, code_b)
        subcode_z = ~ClassicalCode.tensor_product(~code_a, ~code_b)
        code_x = TannerCode(subgraph_x, subcode_x)
        code_z = TannerCode(subgraph_z, subcode_z)
        CSSCode.__init__(self, code_x, code_z, field, conjugate=conjugate, skip_validation=True)

    @classmethod
    def get_subgraphs(cls, cayplex: CayleyComplex) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Build the subgraphs of the inner (classical) Tanner codes for a quantum Tanner code.

        These subgraphs are defined using the faces of a Cayley complex.  Each face looks like:

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

        where f(g,a,b) = {g, ab, gb, agb}.  Specifically, the (directed) subgraphs are:
        - subgraph_x with edges ( g, f(g,a,b)), and
        - subgraph_z with edges (ag, f(g,a,b)).
        Classical Tanner codes on these subgraphs are used as to construct quantum Tanner code.

        As a matter of practice, defining classical Tanner codes on subgraph_x and subgraph_z
        requires choosing an ordering on the edges incident to every source node of these graphs.
        If the group G is equipped with a total order, a natural ordering of edges incident to every
        source node is induced by assigning the label (a, b) to edge (g, f(g,a,b)).  Consistency
        then requires that edge (ag, f(g,a,b)) has label (a^-1, b), as verified by defining g' = ag
        and checking that f(g,a,b) = f(g',a^-1,b).
        """
        subgraph_x = nx.DiGraph()
        subgraph_z = nx.DiGraph()
        nodes_x, _ = nx.bipartite.sets(cayplex.graph)
        for gg, aa, bb in itertools.product(nodes_x, cayplex.subset_a, cayplex.subset_b):
            aa_gg, gg_bb, aa_gg_bb = aa * gg, gg * bb, aa * gg * bb
            face = frozenset([gg, aa_gg, gg_bb, aa_gg_bb])
            subgraph_x.add_edge(gg, face, sort=(aa, bb))
            subgraph_z.add_edge(aa_gg, face, sort=(~aa, bb))
        return subgraph_x, subgraph_z


################################################################################
# common quantum codes


class SurfaceCode(CSSCode):
    """The one and only!

    Actually, there are two variants: "ordinary" and "rotated" surface codes.
    The rotated code is more qubit-efficient.

    If constructed with conjugate=True, every other qubit is Hadamard-transformed in a checkerboard
    pattern.  The rotated surface code with conjugate=True is the XZZX code in arXiv:2009.07851.
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        rotated: bool = True,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        if cols is None:
            cols = rows

        # save known distances
        self._exact_distance_x = cols
        self._exact_distance_z = rows

        # which qubits should be Hadamard-transformed?
        qudits_to_conjugate: slice | Sequence[int] | None

        if rotated:
            # rotated surface code
            matrix_x, matrix_z = SurfaceCode.get_rotated_checks(rows, cols)

            if conjugate:
                # Hadamard-transform qubits in a checkerboard pattern
                qudits_to_conjugate = [
                    idx for idx, (row, col) in enumerate(np.ndindex(rows, cols)) if (row + col) % 2
                ]

            else:
                qudits_to_conjugate = None

        else:
            # "original" surface code
            code_a = RepetitionCode(rows, field)
            code_b = RepetitionCode(cols, field)
            code_ab = HGPCode(code_a, code_b, field, conjugate=conjugate)
            matrix_x = code_ab.matrix_x
            matrix_z = code_ab.matrix_z
            qudits_to_conjugate = code_ab.conjugated_qubits

        CSSCode.__init__(
            self,
            matrix_x,
            matrix_z,
            field=field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )

    @classmethod
    def get_rotated_checks(
        cls, rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Example 5x5 rotated surface code layout:

             ―――     ―――
            | ⋅ |   | ⋅ |
            ○―――○―――○―――○―――○―――
            | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
            | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○
                | ⋅ |   | ⋅ |
                 ―――     ―――

        Here:
        - Circles (○) denote data qubits (of which there are 5×5 = 25 total).
        - Tiles with a cross (×) denote X-type parity checks (12 total).
        - Tiles with a dot (⋅) denote Z-type parity checks (12 total).

        Reference: https://errorcorrectionzoo.org/c/rotated_surface
        """

        def get_check(
            row_indices: Sequence[int], col_indices: Sequence[int]
        ) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, dropping any that are out of bounds."""
            check = np.zeros((rows, cols), dtype=int)
            for row, col in zip(row_indices, col_indices):
                if 0 <= row < rows and 0 <= col < cols:
                    check[row, col] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row in range(-1, rows):
            for col in range(-1, cols):
                row_indices = [row, row + 1, row, row + 1]
                col_indices = [col, col, col + 1, col + 1]
                check = get_check(row_indices, col_indices)

                # exclude exterior corner tiles that only touch one data qubit
                if np.count_nonzero(check) == 1:
                    continue

                if row % 2 == col % 2:
                    if 0 <= row < rows - 1:
                        # no X-type parity checks on the top/bottom boundaries
                        checks_x.append(check)
                elif 0 <= col < cols - 1:
                    # no Z-type parity checks on the left/right boundaries
                    checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)


class ToricCode(CSSCode):
    """Surface code with periodic bounary conditions, encoding two logical qudits.

    Reference: https://errorcorrectionzoo.org/c/surface
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        rotated: bool = True,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        if cols is None:
            cols = rows

        # save known distances
        self._exact_distance_x = self._exact_distance_z = min(rows, cols)

        # which qubits should be Hadamard-transformed?
        qudits_to_conjugate: slice | Sequence[int] | None

        if rotated:
            # rotated toric code
            if not (rows % 2 == cols % 2 == 0 and rows >= 4 and cols >= 4):
                raise ValueError(
                    "The rotated toric code must have even side lengths of at least four, not"
                    + f" ({rows},{cols})"
                )
            matrix_x, matrix_z = ToricCode.get_rotated_checks(rows, cols)

            if conjugate:
                # Hadamard-transform qubits in a checkerboard pattern
                qudits_to_conjugate = [
                    idx for idx, (row, col) in enumerate(np.ndindex(rows, cols)) if (row + col) % 2
                ]

            else:
                qudits_to_conjugate = None

        else:
            # "original" toric code
            code_a = RingCode(rows, field)
            code_b = RingCode(cols, field)
            code_ab = HGPCode(code_a, code_b, field, conjugate=conjugate)
            matrix_x = code_ab.matrix_x
            matrix_z = code_ab.matrix_z
            qudits_to_conjugate = code_ab.conjugated_qubits

        CSSCode.__init__(
            self,
            matrix_x,
            matrix_z,
            field=field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )

    @classmethod
    def get_rotated_checks(
        cls, rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Same as in SurfaceCode.get_rotated_checks, but with periodic boundary conditions.
        """

        def get_check(
            row_indices: Sequence[int], col_indices: Sequence[int]
        ) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, with periodic boundary conditions."""
            check = np.zeros((rows, cols), dtype=int)
            for row, col in zip(row_indices, col_indices):
                check[row % rows, col % cols] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row in range(rows):
            for col in range(cols):
                row_indices = [row, row + 1, row, row + 1]
                col_indices = [col, col, col + 1, col + 1]
                check = get_check(row_indices, col_indices)
                if row % 2 == col % 2:
                    checks_x.append(check)
                else:
                    checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)


class GeneralizedSurfaceCode(CSSCode):
    """Surface or toric code defined on a multi-dimensional hypercubic lattice.

    Reference: https://errorcorrectionzoo.org/c/higher_dimensional_surface
    """

    def __init__(
        self,
        size: int,
        dim: int,
        periodic: bool = False,
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] | None = (),
    ) -> None:
        if dim < 2:
            raise ValueError(
                f"The dimension of a generalized surface code should be >= 2 (provided: {dim})"
            )

        base_code = RingCode(size, field) if periodic else RepetitionCode(size, field)

        # build a chain complex one link at a time
        chain = ChainComplex(base_code.matrix)
        link = ChainComplex(base_code.matrix.T)
        for _ in range(dim - 1):
            chain = ChainComplex.tensor_product(chain, link)

            # to reduce computational overhead, remove chain links that we don't care about
            chain = ChainComplex(*chain.ops[:2])

        matrix_x, matrix_z = chain.op(1), chain.op(2).T
        assert not isinstance(matrix_x, abstract.Protograph)
        assert not isinstance(matrix_z, abstract.Protograph)
        CSSCode.__init__(self, matrix_x, matrix_z, field, conjugate=conjugate, skip_validation=True)
