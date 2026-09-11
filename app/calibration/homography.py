"""Planar homography: the projective transform between two planes.

Pure Python, no numpy. That is not stubbornness — :mod:`app.reasoning` consumes
calibration, and the layering tests assert that importing the reasoning layer
pulls in no model runtime at all. An 8-unknown linear solve does not justify
breaching that boundary.

The maths
---------
A homography maps a point on one plane to a point on another under perspective,
using homogeneous coordinates::

    | X' |   | h0 h1 h2 | | x |
    | Y' | = | h3 h4 h5 | | y |          X = X'/W',  Y = Y'/W'
    | W' |   | h6 h7 h8 | | 1 |

The matrix is defined up to scale, so it has 8 degrees of freedom and needs 4
point correspondences (each giving 2 equations). Fixing ``h8 = 1`` turns the
Direct Linear Transform into an ordinary least-squares problem::

    h0*x + h1*y + h2 - h6*x*X - h7*y*X = X
    h3*x + h4*y + h5 - h6*x*Y - h7*y*Y = Y

Why this is the right model for a fixed camera watching a floor: every point
*on that floor* satisfies it exactly, with no knowledge of focal length, camera
height or tilt required. Its limits follow from the same fact — see
:mod:`app.calibration.planar`.

Conditioning
------------
Raw pixel coordinates (hundreds) and world coordinates (tens) produce a badly
scaled system whose normal equations amplify rounding error. Both point sets
are therefore Hartley-normalised before the solve — translated to the centroid
and scaled so the mean distance from it is sqrt(2) — and the resulting matrix
is de-normalised afterwards. This is standard practice and costs ~20 lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Matrix3 = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]

#: Below this, a homogeneous W component means the point maps to (or beyond)
#: the horizon and has no finite image on the other plane.
_MIN_W = 1e-12


class DegenerateConfigurationError(ValueError):
    """The supplied correspondences cannot determine a homography."""


# ---------------------------------------------------------------------------
# Small dense linear algebra
# ---------------------------------------------------------------------------
def solve_linear_system(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> List[float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Raises :class:`DegenerateConfigurationError` when the system is singular,
    which is how a degenerate point configuration surfaces numerically.
    """
    size = len(vector)
    augmented = [list(map(float, row)) + [float(vector[i])] for i, row in enumerate(matrix)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot_row][column]) < 1e-12:
            raise DegenerateConfigurationError(
                "Singular system: the correspondences do not determine a unique "
                "transform (are the points collinear or duplicated?)"
            )
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        pivot = augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                augmented[row][col] -= factor * augmented[column][col]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for col in range(row + 1, size):
            total -= augmented[row][col] * solution[col]
        solution[row] = total / augmented[row][row]
    return solution


def mat3_multiply(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat3_determinant(m: Matrix3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def mat3_inverse(m: Matrix3) -> Matrix3:
    determinant = mat3_determinant(m)
    if abs(determinant) < 1e-15:
        raise DegenerateConfigurationError(
            "Transform is not invertible; it cannot be reversed."
        )
    cofactors = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]
    return tuple(  # type: ignore[return-value]
        tuple(value / determinant for value in row) for row in cofactors
    )


# ---------------------------------------------------------------------------
# Hartley normalisation
# ---------------------------------------------------------------------------
def normalisation_matrix(points: Sequence[Point]) -> Matrix3:
    """Similarity transform centring ``points`` with mean distance sqrt(2)."""
    count = len(points)
    cx = sum(p[0] for p in points) / count
    cy = sum(p[1] for p in points) / count

    mean_distance = (
        sum(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in points) / count
    )
    if mean_distance < 1e-12:
        raise DegenerateConfigurationError(
            "All supplied points are coincident; they carry no spatial information."
        )
    scale = (2.0**0.5) / mean_distance
    return ((scale, 0.0, -scale * cx), (0.0, scale, -scale * cy), (0.0, 0.0, 1.0))


def apply_matrix(matrix: Matrix3, point: Sequence[float]) -> Point:
    """Apply a 3x3 projective matrix to a 2D point."""
    x, y = float(point[0]), float(point[1])
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denominator) < _MIN_W:
        raise DegenerateConfigurationError(
            f"Point ({x:.2f}, {y:.2f}) maps to the horizon of this transform "
            "and has no finite counterpart."
        )
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
    )


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Homography:
    """A projective transform between two planes.

    Deliberately unit-agnostic: it maps numbers to numbers. Attaching meaning
    to those numbers — "these are metres on the floor" — is
    :class:`~app.calibration.planar.GroundPlaneCalibration`'s job, so that
    units are asserted in exactly one place.
    """

    matrix: Matrix3

    def __post_init__(self) -> None:
        normalised = tuple(tuple(float(v) for v in row) for row in self.matrix)
        if len(normalised) != 3 or any(len(row) != 3 for row in normalised):
            raise ValueError("Homography matrix must be 3x3")
        object.__setattr__(self, "matrix", normalised)

    # ------------------------------------------------------------------
    @classmethod
    def from_correspondences(
        cls, source: Sequence[Point], destination: Sequence[Point]
    ) -> "Homography":
        """Fit a homography mapping ``source`` onto ``destination``.

        Four correspondences give an exact solution; more are fitted by least
        squares. Both sets are normalised before the solve and the result is
        de-normalised, so the answer does not depend on whether the inputs are
        measured in pixels, metres or furlongs.
        """
        if len(source) != len(destination):
            raise ValueError(
                f"Point count mismatch: {len(source)} source vs "
                f"{len(destination)} destination"
            )
        if len(source) < 4:
            raise ValueError(
                f"A homography needs at least 4 correspondences, got {len(source)}"
            )

        source_norm = normalisation_matrix(source)
        destination_norm = normalisation_matrix(destination)
        src = [apply_matrix(source_norm, p) for p in source]
        dst = [apply_matrix(destination_norm, p) for p in destination]

        rows: List[List[float]] = []
        values: List[float] = []
        for (x, y), (X, Y) in zip(src, dst):
            rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * X, -y * X])
            values.append(X)
            rows.append([0.0, 0.0, 0.0, x, y, 1.0, -x * Y, -y * Y])
            values.append(Y)

        h = _least_squares(rows, values)
        normalised_matrix: Matrix3 = (
            (h[0], h[1], h[2]),
            (h[3], h[4], h[5]),
            (h[6], h[7], 1.0),
        )
        # H = T_dst^-1 . H_norm . T_src
        matrix = mat3_multiply(
            mat3_inverse(destination_norm),
            mat3_multiply(normalised_matrix, source_norm),
        )
        # Scale so the bottom-right entry is 1 where possible, purely so that
        # two equal transforms compare equal when printed.
        if abs(matrix[2][2]) > 1e-12:
            factor = 1.0 / matrix[2][2]
            matrix = tuple(  # type: ignore[assignment]
                tuple(value * factor for value in row) for row in matrix
            )
        return cls(matrix=matrix)

    # ------------------------------------------------------------------
    def apply(self, point: Sequence[float]) -> Point:
        """Map one point through the transform."""
        return apply_matrix(self.matrix, point)

    def apply_many(self, points: Sequence[Sequence[float]]) -> List[Point]:
        return [self.apply(p) for p in points]

    def inverse(self) -> "Homography":
        """The transform in the opposite direction."""
        return Homography(matrix=mat3_inverse(self.matrix))

    @property
    def determinant(self) -> float:
        return mat3_determinant(self.matrix)

    def to_list(self) -> List[List[float]]:
        return [list(row) for row in self.matrix]

    def to_dict(self) -> Dict[str, Any]:
        return {"matrix": self.to_list(), "determinant": self.determinant}


def _least_squares(rows: Sequence[Sequence[float]], values: Sequence[float]) -> List[float]:
    """Solve an over-determined system via the normal equations.

    ``A^T A h = A^T b``. Squaring the condition number would be a real concern
    on raw coordinates; on Hartley-normalised input it is not, which is the
    reason normalisation is not optional here.
    """
    width = len(rows[0])
    ata = [[0.0] * width for _ in range(width)]
    atb = [0.0] * width
    for row, value in zip(rows, values):
        for i in range(width):
            atb[i] += row[i] * value
            for j in range(width):
                ata[i][j] += row[i] * row[j]
    return solve_linear_system(ata, atb)
