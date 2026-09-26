from __future__ import annotations

import pandas as pd

from scripts.run_qref_decomposition import factorial_blocks, variant_specs


def test_variant_specs_cross_catalog_and_mapping_axes() -> None:
    specs = list(variant_specs(["a", "b", "c"]))
    assert len(specs) == 10
    assert ("pooled_pool_pooled_map", "pooled", "pooled") in specs
    assert ("family_pool_pooled_map", "a", "pooled") in specs
    assert ("pooled_pool_family_map", "pooled", "b") in specs
    assert ("family_pool_family_map", "c", "c") in specs


def test_factorial_blocks_compute_declared_interactions() -> None:
    rows = []
    volumes = {"B0_P0": 9, "B0_P1": 9, "B1_P0": 9, "B1_P1": 19}
    depths = {"B0_P0": 4.0, "B0_P1": 4.0, "B1_P0": 4.0, "B1_P1": 2.0}
    for cell in volumes:
        rows.append(
            {
                "variant": "pooled_pool_pooled_map",
                "catalog_family": "pooled",
                "mapping_family": "pooled",
                "community": "test",
                "post_id": "p1",
                "seed": 0,
                "cell": cell,
                "comment_volume": volumes[cell],
                "mean_leaf_depth": depths[cell],
            }
        )
    blocks = factorial_blocks(pd.DataFrame(rows))
    assert len(blocks) == 1
    assert blocks.iloc[0]["volume_log_interaction"] > 0
    assert blocks.iloc[0]["depth_interaction"] == -2.0

