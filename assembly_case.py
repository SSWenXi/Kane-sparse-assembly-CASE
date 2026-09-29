"""Self-contained topology-aware generalized matrix assembly case.

The implementation mirrors the public algorithmic boundary described in the
manuscript without importing Pywind or using turbine-specific input data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class AssemblyCase:
    segments: int
    internal_modes: int
    branches: int
    upstream_dofs: int
    global_extra_dofs: int
    global_dofs: int
    local_dofs: int
    segment_dofs: int
    active_counts: np.ndarray
    active_dofs: np.ndarray
    packed_v: np.ndarray
    local_mass: np.ndarray
    local_force: np.ndarray


def _spd_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    factor = rng.standard_normal((size, size))
    matrix = factor.T @ factor
    matrix += np.diag(np.linspace(1.0, 2.0, size))
    return np.ascontiguousarray(matrix)


def build_case(
    segments: int,
    internal_modes: int = 0,
    *,
    branches: int = 3,
    upstream_dofs: int = 20,
    global_extra_dofs: int = 2,
    seed: int = 20260828,
) -> AssemblyCase:
    """Create a deterministic three-branch synthetic assembly problem.

    The dimensions reproduce the manuscript storage counts. For example,
    ``segments=10`` and ``internal_modes=0`` give 202 global coordinates,
    80 reachable coordinates at a blade tip, and local dimension 12.
    """
    if segments < 1:
        raise ValueError("segments must be positive")
    if internal_modes < 0:
        raise ValueError("internal_modes must be non-negative")
    if branches < 1 or upstream_dofs < 1 or global_extra_dofs < 0:
        raise ValueError("invalid topology dimensions")

    segment_dofs = 6 + internal_modes
    local_dofs = 6 + segment_dofs
    max_active = upstream_dofs + segments * segment_dofs
    global_dofs = (
        upstream_dofs
        + branches * segments * segment_dofs
        + global_extra_dofs
    )

    active_counts = np.empty((branches, segments), dtype=np.int32)
    active_dofs = np.full((branches, max_active), -1, dtype=np.int32)
    packed_v = np.zeros(
        (branches, segments, local_dofs, max_active), dtype=np.float64
    )
    local_mass = np.empty(
        (branches, segments, local_dofs, local_dofs), dtype=np.float64
    )
    local_force = np.empty(
        (branches, segments, local_dofs), dtype=np.float64
    )

    rng = np.random.default_rng(seed + 1009 * segments + 9176 * internal_modes)
    upstream = np.arange(upstream_dofs, dtype=np.int32)

    for branch in range(branches):
        branch_start = upstream_dofs + branch * segments * segment_dofs
        branch_coordinates = np.arange(
            branch_start,
            branch_start + segments * segment_dofs,
            dtype=np.int32,
        )
        ordered = np.concatenate((upstream, branch_coordinates))
        active_dofs[branch, :ordered.size] = ordered

        for segment in range(segments):
            active_count = upstream_dofs + (segment + 1) * segment_dofs
            rigid_count = active_count - segment_dofs
            active_counts[branch, segment] = active_count

            # Inherited/upstream coordinates act through the first six rows.
            rigid_block = rng.standard_normal((6, rigid_count))
            rigid_block *= 0.2 / np.sqrt(max(rigid_count, 1))
            rigid_block[:, :6] += np.eye(6)
            packed_v[branch, segment, :6, :rigid_count] = rigid_block

            # The segment's newly introduced coordinates occupy the local
            # interface/internal rows and the final columns of the active set.
            local_block = rng.standard_normal((segment_dofs, segment_dofs))
            local_block *= 0.05
            local_block += np.eye(segment_dofs)
            packed_v[
                branch,
                segment,
                6:,
                rigid_count:active_count,
            ] = local_block

            local_mass[branch, segment] = _spd_matrix(rng, local_dofs)
            local_force[branch, segment] = rng.standard_normal(local_dofs)

    return AssemblyCase(
        segments=segments,
        internal_modes=internal_modes,
        branches=branches,
        upstream_dofs=upstream_dofs,
        global_extra_dofs=global_extra_dofs,
        global_dofs=global_dofs,
        local_dofs=local_dofs,
        segment_dofs=segment_dofs,
        active_counts=active_counts,
        active_dofs=active_dofs,
        packed_v=np.ascontiguousarray(packed_v),
        local_mass=np.ascontiguousarray(local_mass),
        local_force=np.ascontiguousarray(local_force),
    )


def materialize_dense_v(case: AssemblyCase) -> np.ndarray:
    """Expand packed partial velocities to global zero-padded columns."""
    dense = np.zeros(
        (
            case.branches,
            case.segments,
            case.local_dofs,
            case.global_dofs,
        ),
        dtype=np.float64,
    )
    for branch in range(case.branches):
        for segment in range(case.segments):
            count = int(case.active_counts[branch, segment])
            dofs = case.active_dofs[branch, :count]
            dense[branch, segment][:, dofs] = case.packed_v[
                branch, segment, :, :count
            ]
    return np.ascontiguousarray(dense)


def assemble_dense(case: AssemblyCase, dense_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Algebraic reference using full global-column matrices."""
    generalized_mass = np.zeros((case.global_dofs, case.global_dofs))
    generalized_force = np.zeros(case.global_dofs)
    for branch in range(case.branches):
        for segment in range(case.segments):
            velocity = dense_v[branch, segment]
            generalized_mass += (
                velocity.T @ case.local_mass[branch, segment] @ velocity
            )
            generalized_force += velocity.T @ case.local_force[branch, segment]
    return generalized_mass, generalized_force


@njit(cache=True)
def _assemble_packed_kernel(
    packed_v: np.ndarray,
    active_dofs: np.ndarray,
    active_counts: np.ndarray,
    local_mass: np.ndarray,
    local_force: np.ndarray,
    global_dofs: int,
) -> tuple[np.ndarray, np.ndarray]:
    branches, segments, local_dofs, max_active = packed_v.shape
    segment_dofs = local_dofs - 6
    generalized_mass = np.zeros((global_dofs, global_dofs))
    generalized_force = np.zeros(global_dofs)
    workspace = np.zeros((local_dofs, max_active))

    for branch in range(branches):
        for segment in range(segments):
            active_count = active_counts[branch, segment]
            rigid_count = active_count - segment_dofs
            velocity = packed_v[branch, segment]
            mass = local_mass[branch, segment]
            force = local_force[branch, segment]
            workspace[:, :active_count] = 0.0

            # W[:, a] = M @ V[:, a]. Inherited columns have support only in
            # rows 0:6; newly introduced segment columns only in rows 6:p.
            for a in range(rigid_count):
                for row in range(local_dofs):
                    value = 0.0
                    for j in range(6):
                        value += mass[row, j] * velocity[j, a]
                    workspace[row, a] = value

            for a in range(rigid_count, active_count):
                for row in range(local_dofs):
                    value = 0.0
                    for j in range(6, local_dofs):
                        value += mass[row, j] * velocity[j, a]
                    workspace[row, a] = value

            # Compute only the upper triangle and mirror it during scatter.
            for a in range(rigid_count):
                global_a = active_dofs[branch, a]
                force_value = 0.0
                for row in range(6):
                    force_value += velocity[row, a] * force[row]
                generalized_force[global_a] += force_value

                for b in range(a, active_count):
                    global_b = active_dofs[branch, b]
                    value = 0.0
                    for row in range(6):
                        value += velocity[row, a] * workspace[row, b]
                    generalized_mass[global_a, global_b] += value
                    if global_a != global_b:
                        generalized_mass[global_b, global_a] += value

            for a in range(rigid_count, active_count):
                global_a = active_dofs[branch, a]
                force_value = 0.0
                for row in range(6, local_dofs):
                    force_value += velocity[row, a] * force[row]
                generalized_force[global_a] += force_value

                for b in range(a, active_count):
                    global_b = active_dofs[branch, b]
                    value = 0.0
                    for row in range(6, local_dofs):
                        value += velocity[row, a] * workspace[row, b]
                    generalized_mass[global_a, global_b] += value
                    if global_a != global_b:
                        generalized_mass[global_b, global_a] += value

    return generalized_mass, generalized_force


def assemble_packed(case: AssemblyCase) -> tuple[np.ndarray, np.ndarray]:
    """Topology-aware assembly using compact reachable-coordinate columns."""
    return _assemble_packed_kernel(
        case.packed_v,
        case.active_dofs,
        case.active_counts,
        case.local_mass,
        case.local_force,
        case.global_dofs,
    )


def storage_bytes(case: AssemblyCase, dense_v: np.ndarray) -> dict[str, int]:
    """Return deterministic partial-velocity storage, excluding output G/f."""
    packed = case.packed_v.nbytes + case.active_dofs.nbytes + case.active_counts.nbytes
    return {
        "packed_v_only": int(case.packed_v.nbytes),
        "packed_with_indices": int(packed),
        "dense_v": int(dense_v.nbytes),
    }


def validate_topology(case: AssemblyCase) -> None:
    """Raise AssertionError if active sets or structural-zero rows are invalid."""
    expected = case.upstream_dofs + (
        np.arange(1, case.segments + 1) * case.segment_dofs
    )
    for branch in range(case.branches):
        np.testing.assert_array_equal(case.active_counts[branch], expected)
        previous: set[int] = set()
        for segment in range(case.segments):
            count = int(case.active_counts[branch, segment])
            current = set(map(int, case.active_dofs[branch, :count]))
            assert previous.issubset(current)
            previous = current

            rigid_count = count - case.segment_dofs
            velocity = case.packed_v[branch, segment]
            assert np.count_nonzero(velocity[6:, :rigid_count]) == 0
            assert np.count_nonzero(velocity[:6, rigid_count:count]) == 0
