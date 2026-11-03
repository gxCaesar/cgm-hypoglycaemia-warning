from pathlib import Path

import pytest

from glycoshift.config import load_config
from glycoshift.run_seed_suite import seeded_configs, select_seeded_configs


def test_seed_suite_changes_training_seed_but_not_split(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["training"]["final_seeds"] = [17, 23]
    config["output"]["run_dir"] = str(tmp_path / "suite")
    configs = seeded_configs(config, model_name="glycoshift_tiny")
    assert [item["training"]["seed"] for item in configs] == [17, 23]
    assert {item["split"]["seed"] for item in configs} == {42}
    assert configs[0]["output"]["run_dir"].endswith(
        "glycoshift_tiny-seed-017"
    )


def test_real_seed_suite_binds_preprocessing_artifacts(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["project"]["synthetic"] = False
    config["training"]["final_seeds"] = [17]
    preprocessing_dir = tmp_path / "preprocessing"
    config["output"]["run_dir"] = str(preprocessing_dir)

    run_config = seeded_configs(
        config,
        model_name="glycoshift_base",
    )[0]

    assert run_config["data"]["training_window_shards"] == str(
        preprocessing_dir / "window_shards"
    )
    assert run_config["data"]["training_cohort_audit"] == str(
        preprocessing_dir / "cohort_audit.json"
    )
    assert run_config["output"]["run_dir"] == str(
        preprocessing_dir / "glycoshift_base-seed-017"
    )


def test_seed_suite_selects_only_requested_frozen_seeds(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["training"]["final_seeds"] = [17, 23, 42, 71, 101]
    config["output"]["run_dir"] = str(tmp_path / "suite")

    configs = select_seeded_configs(
        config,
        model_name="glycoshift_base",
        seeds=[17, 42, 101],
    )

    assert [item["training"]["seed"] for item in configs] == [17, 42, 101]
    assert {item["split"]["seed"] for item in configs} == {42}


def test_seed_suite_rejects_nonfrozen_or_duplicate_requested_seed(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["training"]["final_seeds"] = [17, 23]
    config["output"]["run_dir"] = str(tmp_path / "suite")

    with pytest.raises(ValueError, match="not in training.final_seeds"):
        select_seeded_configs(
            config,
            model_name="glycoshift_base",
            seeds=[42],
        )
    with pytest.raises(ValueError, match="contain duplicates"):
        select_seeded_configs(
            config,
            model_name="glycoshift_base",
            seeds=[17, 17],
        )
