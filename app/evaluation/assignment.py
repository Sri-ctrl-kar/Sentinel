"""Optimal assignment (Hungarian / Kuhn-Munkres).

Why this exists
---------------
MOTA and IDF1 are defined in terms of an *optimal* matching — MOTA matches
detections to ground truth per frame, IDF1 matches predicted identities to
ground-truth identities over the whole sequence. Greedy matching gives
different, generally better-looking numbers, so reporting greedy results under
those names would be quietly wrong.

This is the standard O(n^3) shortest-augmenting-path formulation with
potentials. Pure Python, no numpy: the evaluation layer is held to the same
no-model-runtime rule as the reasoning layer.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

INF = float("inf")


def solve_min_cost(cost: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Minimum-cost assignment of rows to columns.

    Returns ``[(row, column), ...]`` covering ``min(rows, columns)`` pairs.
    Rectangular inputs are handled by padding internally, so no caller has to
    square the matrix itself.
    """
    rows = len(cost)
    if rows == 0:
        return []
    columns = len(cost[0])
    if columns == 0:
        return []

    transposed = rows > columns
    matrix = _transpose(cost) if transposed else [list(row) for row in cost]
    n, m = len(matrix), len(matrix[0])

    # e-maxx formulation: 1-indexed, with a virtual row 0 and column 0.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    assigned_row_of_column = [0] * (m + 1)
    path = [0] * (m + 1)

    for i in range(1, n + 1):
        assigned_row_of_column[0] = i
        j0 = 0
        minimum = [INF] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = assigned_row_of_column[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    path[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[assigned_row_of_column[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if assigned_row_of_column[j0] == 0:
                break

        while j0:
            j1 = path[j0]
            assigned_row_of_column[j0] = assigned_row_of_column[j1]
            j0 = j1

    pairs = [
        (assigned_row_of_column[j] - 1, j - 1)
        for j in range(1, m + 1)
        if assigned_row_of_column[j] > 0
    ]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return sorted(pairs)


def solve_max_score(
    score: Sequence[Sequence[float]], minimum_score: float = 0.0
) -> List[Tuple[int, int]]:
    """Maximum-score assignment, dropping pairs scoring at or below the floor.

    Convenience wrapper for affinity matrices (IoU, overlap counts): maximising
    a score is minimising its negation.
    """
    if not score or not score[0]:
        return []
    cost = [[-value for value in row] for row in score]
    return [
        (row, column)
        for row, column in solve_min_cost(cost)
        if score[row][column] > minimum_score
    ]


def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(column) for column in zip(*matrix)]
