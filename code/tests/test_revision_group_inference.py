"""Synthetic complete inventories and exact, private pair bindings only."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_revision_external import _binding
from test_revision_external_inference import (
    SOURCE,
    _accepted_fixture,
    _joint_fixture,
)

from glycoshift import revision_external_inference as inference
from glycoshift.revision_group_inference import (
    evaluation_keys,
    exclude_pair_records,
    load_external_record_groups,
    validated_horizon_counts,
)


def _group_files(tmp_path, *, freeze_id="toy", source=None, records=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    if source is None:
        normalized = tmp_path / "normalized-metadata-only"
        normalized.write_text("synthetic normalized binding; contents never consumed")
        source = {"source_id": SOURCE, "normalized_inputs": [_binding(normalized)]}
    records = records or ("p0", "p1", "zero")
    identity = {
        "schema_version": "1.0", "freeze_id": freeze_id,
        "partition": "external_loop", "source_id": SOURCE,
    }
    inventory_path = tmp_path / "records.private.json"
    inventory_path.write_text(json.dumps({
        **identity, "normalized_inputs": source["normalized_inputs"],
        "records": [
            {"source_id": SOURCE, "participant_id": record} for record in records
        ],
    }))
    pair_path = tmp_path / "pair.private.json"
    pair_path.write_text(json.dumps({
        **identity, "record_manifest": _binding(inventory_path),
        "status": "verified_partial_overlap",
        "members": [{"source_id": SOURCE, "participant_id": record}
                    for record in ("p0", "p1")],
    }))
    group_path = tmp_path / "groups.private.json"
    group_path.write_text(json.dumps({
        **identity, "record_manifest": _binding(inventory_path),
        "pair_manifest": _binding(pair_path),
        "groups": [
            {"source_id": SOURCE, "participant_id": record,
             "group_id": "pair" if record in ("p0", "p1") else record}
            for record in records
        ],
    }))
    return group_path, source


def _load(path, source, count=3):
    return load_external_record_groups(
        path, freeze_id="toy", partition="external_loop", source=source,
        normalized_participants=count,
    )


def test_full_inventory_is_validated_before_restricting_to_exposed_records(tmp_path):
    path, source = _group_files(tmp_path, records=("p0", "p1", "zero", "no-exposure"))
    groups = _load(path, source, 4)
    mapping = groups.restrict(evaluation_keys(_joint_fixture()))
    assert len(mapping) == 3
    assert groups.aggregate_counts()["full_record_count"] == 4
    assert groups.aggregate_counts()["full_group_count"] == 3
    assert len(set(mapping.values())) == 2
    with pytest.raises(ValueError, match="outside the complete record inventory"):
        groups.restrict({(SOURCE, "not-in-inventory")})
    with pytest.raises(ValueError, match="native source population"):
        _load(path, source, 3)


@pytest.mark.parametrize("defect", [
    "missing_record", "extra_record", "duplicate_record", "mixed_id", "pair_status",
    "pair_binding", "wrong_pair", "extra_merge", "split_pair", "source", "freeze",
    "normalized_binding", "changed_file",
])
def test_known_bad_inventory_and_verified_pair_bindings_fail_closed(tmp_path, defect):
    path, source = _group_files(tmp_path)
    groups = json.loads(path.read_text())
    if defect in ("pair_status", "pair_binding", "wrong_pair", "changed_file"):
        pair_path = tmp_path / "pair.private.json"
        pair = json.loads(pair_path.read_text())
        if defect == "pair_status":
            pair["status"] = "suspected"
        elif defect == "pair_binding":
            pair["record_manifest"] = _binding(path)
        elif defect == "wrong_pair":
            pair["members"][1]["participant_id"] = "foreign"
        pair_path.write_text(json.dumps(pair) + " ")
        if defect != "changed_file":
            groups["pair_manifest"] = _binding(pair_path)
    elif defect == "missing_record":
        groups["groups"].pop()
    elif defect == "extra_record":
        groups["groups"].append({
            "source_id": SOURCE, "participant_id": "extra", "group_id": "extra",
        })
    elif defect == "duplicate_record":
        groups["groups"].append(groups["groups"][0])
    elif defect == "mixed_id":
        groups["groups"][0]["participant_id"] = 7
    elif defect == "extra_merge":
        groups["groups"][2]["group_id"] = "pair"
    elif defect == "split_pair":
        groups["groups"][1]["group_id"] = "separate"
    elif defect == "source":
        groups["groups"][0]["source_id"] = "foreign"
    elif defect == "freeze":
        groups["freeze_id"] = "another-freeze"
    else:
        inventory_path = tmp_path / "records.private.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["normalized_inputs"] = []
        inventory_path.write_text(json.dumps(inventory))
        groups["record_manifest"] = _binding(inventory_path)
    path.write_text(json.dumps(groups))
    with pytest.raises(ValueError):
        _load(path, source)


def test_cluster_inference_keeps_record_weights_and_horizon_groups(
    tmp_path, monkeypatch,
):
    path, source = _group_files(tmp_path)
    groups = _load(path, source)
    values = _joint_fixture()

    class OracleDraw:
        def __init__(self):
            self.draws = iter(([0, 0], [1, 1], [0, 1]))

        def choice(self, options, *, size, replace):
            assert size == len(options) and replace
            return options.copy() if size == 1 else options[next(self.draws)]

    monkeypatch.setattr(np.random, "default_rng", lambda _: OracleDraw())
    result = inference.paired_external_inference(
        values, source_order=(SOURCE,), model_seed_order=(17, 23), replicates=3,
        participant_groups=groups.restrict(evaluation_keys(values)),
        include_replicates=True,
    )
    assert {row["participant_count"] for row in result["counts"]} == {3}
    assert {row["group_count"] for row in result["counts"]} == {2}
    sensitivity = result["metrics"]["fixed_sensitivity"]
    assert sensitivity["interval_status"] == "descriptive_sparse_event_positive_groups"
    assert sensitivity["contrasts"]["lstm_at_54"]["estimate"] == 0.5
    assert sensitivity["contrasts"]["lstm_at_54"]["ci_low"] is None
    burden = result["metrics"]["delivered_alarms_per_patient_day"]
    np.testing.assert_allclose(
        burden["contrasts"]["lstm_at_54"]["replicate_estimates"],
        # Pair draws carry both distinct records. Their mean is 1/2; drawing
        # both groups once gives the original record-weighted mean of 1/3.
        [1 / 2, 0, 1 / 3],
    )


def test_pair_exclusion_rebuilds_source_summaries_without_refitting(tmp_path):
    path, source = _group_files(tmp_path)
    groups = _load(path, source)
    values = _joint_fixture()
    excluded = exclude_pair_records(values, groups.pair)
    for key, value in excluded.items():
        assert value.threshold == values[key].threshold
        assert list(value.participants.participant_id) == ["zero"]
        assert value.events.empty
        assert value.sources.participant_count.tolist() == [1]
        assert value.summary["participant_count"] == 1
        assert not value.summary["sensitivity_defined"]
        assert np.isnan(value.summary[
            "equal_source_participant_macro_fixed_sensitivity"
        ])
        expected = 1 if key.arm == "composition" else 0
        assert value.sources.iloc[0][
            "participant_macro_delivered_alarms_per_patient_day"
        ] == expected


def test_group_consumer_accepts_horizon_presence_changes_but_not_arm_record_loss(
    tmp_path,
):
    path, source = _group_files(tmp_path)
    groups = _load(path, source)
    values = _joint_fixture()
    for key, value in list(values.items()):
        if key.horizon_minutes == 60:
            values[key] = replace(
                value, participants=value.participants.loc[
                    value.participants.participant_id.ne("p1")
                ].copy(), events=value.events.loc[value.events.participant_id.ne("p1")],
            )
    result = inference.paired_external_inference(
        values, source_order=(SOURCE,), model_seed_order=(17, 23), replicates=5,
        participant_groups=groups.restrict(evaluation_keys(values)),
        allow_horizon_population_difference=True,
    )
    assert [row["participant_count"] for row in result["counts"]] == [3, 2]
    assert [row["group_count"] for row in result["counts"]] == [2, 2]
    key = next(key for key in values if key.arm == "lstm" and key.horizon_minutes == 60)
    values[key] = replace(values[key], participants=values[key].participants.iloc[:1])
    with pytest.raises(ValueError, match="same-horizon bootstrap participant rows"):
        inference.paired_external_inference(
            values, source_order=(SOURCE,), model_seed_order=(17, 23), replicates=5,
            participant_groups=groups.restrict(evaluation_keys(values)),
            allow_horizon_population_difference=True,
        )


def test_native_grouped_export_has_named_pair_exclusion_and_no_private_ids(tmp_path):
    freeze_path, producers = _accepted_fixture(
        tmp_path / "native",
        horizon_members={30: ("p0", "p1", "zero"), 60: ("p0", "zero")},
    )
    freeze = json.loads(freeze_path.read_text())
    path, _ = _group_files(
        tmp_path / "private-inputs", freeze_id=freeze["freeze_id"],
        source=freeze["partitions"]["external_loop"],
    )
    output = inference.run_external_paired_inference(
        freeze_path, producers=producers, partition="external_loop",
        output_dir=tmp_path / "aggregates", group_manifest_path=path,
    )
    report = json.loads((output / "external-paired-inference.json").read_text())
    assert report["correlation_handling"]["full_record_count"] == 3
    assert report["correlation_handling"]["full_group_count"] == 2
    excluded = report["named_sensitivities"]["exclude_verified_pair_records"]
    assert excluded["thresholds_refit"] is excluded["fitted_on_external"] is False
    assert {row["participant_count"] for row in excluded["counts"]} == {1}
    assert all(
        row["participant_count"] == 1 for row in excluded["descriptive_policies"]
    )
    assert [row["participant_count"] for row in report["counts"]] == [3, 2]
    text = (output / "external-paired-inference.json").read_text()
    assert all(word not in text for word in (
        "participant_id", '"p0"', '"p1"', '"zero"',
    ))
    csv = pd.read_csv(output / "external-paired-inference.csv")
    assert set(csv.analysis) == {"primary", "exclude_verified_pair_records"}


@pytest.mark.parametrize("bad", [
    {"30": {"evaluated_participants": 2, "zero_base_exposure_participants": 0}},
    {str(h): {"evaluated_participants": 2, "zero_base_exposure_participants": 0}
     for h in (30, 60)},
    {str(h): {"evaluated_participants": True, "zero_base_exposure_participants": 2}
     for h in (30, 60)},
])
def test_bad_horizon_counts_fail_closed(bad):
    with pytest.raises(ValueError, match="horizon participant counts"):
        validated_horizon_counts(bad, 3)
