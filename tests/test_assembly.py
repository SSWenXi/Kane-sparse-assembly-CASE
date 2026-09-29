from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assembly_case import (  # noqa: E402
    assemble_dense,
    assemble_packed,
    build_case,
    materialize_dense_v,
    storage_bytes,
    validate_topology,
)


@pytest.mark.parametrize(
    "segments,internal_modes",
    [(5, 0), (10, 0), (25, 0), (50, 0), (5, 3), (10, 3)],
)
def test_dense_and_packed_assembly_are_equivalent(segments, internal_modes):
    case = build_case(segments, internal_modes)
    validate_topology(case)
    dense_v = materialize_dense_v(case)
    dense_mass, dense_force = assemble_dense(case, dense_v)
    packed_mass, packed_force = assemble_packed(case)

    np.testing.assert_allclose(packed_mass, dense_mass, rtol=2e-13, atol=1e-9)
    np.testing.assert_allclose(packed_force, dense_force, rtol=2e-13, atol=1e-10)
    np.testing.assert_allclose(packed_mass, packed_mass.T, rtol=0.0, atol=1e-11)


def test_manuscript_dimensions_and_storage_for_ten_segments():
    case = build_case(10, 0)
    dense_v = materialize_dense_v(case)
    memory = storage_bytes(case, dense_v)

    assert case.global_dofs == 202
    assert case.local_dofs == 12
    assert int(case.active_counts[0, -1]) == 80
    assert memory["packed_v_only"] == 3 * 10 * 12 * 80 * 8
    assert memory["dense_v"] == 3 * 10 * 12 * 202 * 8
    assert memory["packed_v_only"] < memory["dense_v"]


def test_internal_modes_change_local_and_global_dimensions():
    case = build_case(10, 3)
    assert case.segment_dofs == 9
    assert case.local_dofs == 15
    assert case.global_dofs == 292
    assert int(case.active_counts[0, -1]) == 110

