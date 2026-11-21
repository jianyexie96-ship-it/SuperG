import copy
import json
from types import SimpleNamespace

import pytest
import torch


def configuration(seed=1):
    return {
        "dimension": 768,
        "data": {"protocol": "item_cold", "seed": 1, "exclude_query_item": True},
        "encoder": {"layers": 1},
        "retrieval": {"top_behaviors": 5},
        "contrastive": {"temperature": 0.1},
        "llm": {"model": "fixture", "revision": "fixed"},
        "training": {"seed": seed, "epochs": 2},
        "evaluation": {"ks": [10, 20], "sampled_negatives": None},
        "ablations": {"no_ibmm_score": "semantic", "no_bhrm_adaptation": "zero_pad"},
        "sweeps": {},
    }


def metric_file(path, config):
    report = {
        "config": config,
        "dataset_sha256": "same-data",
        "feature_sha256": "same-features",
        "ablation": "full",
        "protocol": "item_cold",
        "candidate_mode": "full_split_catalog",
        "sampled_negatives": None,
        "split": "test",
        "Recall@10": 0.1,
        "Recall@20": 0.2,
        "NDCG@10": 0.1,
        "NDCG@20": 0.2,
    }
    path.write_text(json.dumps(report))
    return path


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("data", "exclude_query_item", False),
        ("ablations", "no_ibmm_score", "personalized"),
        ("evaluation", "ks", [5, 10, 20]),
    ],
)
def test_aggregation_separates_leakage_ablation_and_ranking_protocols(
    tmp_path, section, key, value
):
    from evaluate import aggregate

    first, second = configuration(), configuration(2)
    second[section][key] = value
    paths = [
        metric_file(tmp_path / "one.json", first),
        metric_file(tmp_path / "two.json", second),
    ]
    assert len(aggregate(paths, tmp_path / "summary.json")) == 2


def test_aggregation_combines_only_training_seed_changes(tmp_path):
    from evaluate import aggregate

    paths = [
        metric_file(tmp_path / f"{seed}.json", configuration(seed)) for seed in (1, 2)
    ]
    rows = aggregate(paths, tmp_path / "summary.json")
    assert len(rows) == 1
    assert sorted(rows[0]["seeds"]) == [1, 2]


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_catalog_scoring_rejects_nonfinite_or_out_of_range_probabilities(
    tmp_path, probability
):
    from evaluate import score_catalog

    data_file, features_file = tmp_path / "data.json", tmp_path / "features.npz"
    data_file.write_text("{}")
    features_file.write_bytes(b"software fixture")
    dataset = SimpleNamespace(
        eligible_users=lambda split: [0],
        item_pools={"train": [0], "val": [1], "test": [1]},
        positives={"train": {0: {0}}, "val": {0: {1}}, "test": {0: {1}}},
        users=["u"],
        metadata={"protocol": "item_cold"},
        path=data_file,
        feature_path=features_file,
    )
    model = SimpleNamespace(
        config={
            "evaluation": {"ks": [1]},
            "retrieval": {"top_behaviors": 1},
            "ablations": {},
        },
        eval=lambda: None,
        context=lambda *args: {"behaviors": [], "scores": torch.empty(0)},
        retriever=SimpleNamespace(rank=lambda *args: []),
        prompt=lambda *args: ("fixture", torch.empty(0, 1)),
        llm=SimpleNamespace(
            score=lambda *args, **kwargs: {"p_yes": torch.tensor([probability])}
        ),
    )
    with pytest.raises(ValueError, match="probab|score|finite"):
        score_catalog(
            model, dataset, SimpleNamespace(refresh=lambda: None), torch.zeros(2, 1)
        )


def test_sweep_skips_verified_completed_training_and_resumes_partial_jobs(
    tmp_path, monkeypatch
):
    import main
    from data import fingerprint, write_json

    data_file, features_file = tmp_path / "data.json", tmp_path / "features.npz"
    data_file.write_text("{}")
    features_file.write_bytes(b"software fixture")
    root = tmp_path / "runs"
    complete = root / "full" / "seed_1" / "checkpoints"
    partial = root / "full" / "seed_2" / "checkpoints"
    complete.mkdir(parents=True)
    partial.mkdir(parents=True)
    (complete / "best.pt").write_bytes(b"complete software fixture")
    (partial / "last.pt").write_bytes(b"partial software fixture")
    write_json(complete / "config.json", configuration())
    write_json(
        complete / "training_complete.json",
        {
            "ablation": "full",
            "best_checkpoint_sha256": fingerprint(complete / "best.pt"),
            "dataset_sha256": fingerprint(data_file),
            "features_sha256": fingerprint(features_file),
            "config_sha256": fingerprint(complete / "config.json"),
        },
    )
    calls = []
    monkeypatch.setattr(
        main.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    args = SimpleNamespace(
        output=root, seeds=[1, 2], data=data_file, features=features_file, execute=True
    )
    main.sweep_jobs(args, configuration())
    assert not any("train" in call and str(complete) in call for call in calls)
    resumed = [call for call in calls if "train" in call and str(partial) in call]
    assert len(resumed) == 1 and "--resume" in resumed[0]
    assert str(partial / "last.pt") in resumed[0]
    assert any(
        "evaluate" in call and str(complete / "best.pt") in call for call in calls
    )


def test_sweep_rejects_a_corrupted_completion_marker(tmp_path, monkeypatch):
    import main
    from data import fingerprint, write_json

    data_file, features_file = tmp_path / "data.json", tmp_path / "features.npz"
    data_file.write_text("{}")
    features_file.write_bytes(b"software fixture")
    root = tmp_path / "runs"
    complete = root / "full" / "seed_1" / "checkpoints"
    complete.mkdir(parents=True)
    (complete / "best.pt").write_bytes(b"modified")
    write_json(complete / "config.json", configuration())
    write_json(
        complete / "training_complete.json",
        {
            "ablation": "full",
            "best_checkpoint_sha256": "old",
            "dataset_sha256": fingerprint(data_file),
            "features_sha256": fingerprint(features_file),
            "config_sha256": fingerprint(complete / "config.json"),
        },
    )
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "Corrupt completed artifacts must not start a subprocess"
        ),
    )
    with pytest.raises(ValueError, match="completion|checkpoint|hash"):
        main.sweep_jobs(
            SimpleNamespace(
                output=root,
                seeds=[1],
                data=data_file,
                features=features_file,
                execute=True,
            ),
            configuration(),
        )


def test_checkpoint_captures_and_restores_mps_rng_without_training(
    tmp_path, monkeypatch
):
    import train

    data_file, features_file = tmp_path / "data.json", tmp_path / "features.npz"
    data_file.write_text("{}")
    features_file.write_bytes(b"software fixture")
    expected = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored, payloads = [], []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: expected)
    monkeypatch.setattr(
        torch.mps, "set_rng_state", lambda value: restored.append(value)
    )

    def save(value, path):
        payloads.append(copy.deepcopy(value))
        path.write_bytes(b"software checkpoint fixture")

    monkeypatch.setattr(torch, "save", save)
    model = SimpleNamespace(
        trainable_state=lambda: {"fixture": True},
        llm=SimpleNamespace(metadata=lambda: {"model": "fixture"}),
    )
    optimizer = SimpleNamespace(state_dict=lambda: {"fixture": True})
    train.save_checkpoint(
        tmp_path / "last.pt",
        model,
        optimizer,
        1,
        1,
        0.1,
        0,
        SimpleNamespace(path=data_file, feature_path=features_file),
        configuration(),
        [],
    )
    assert torch.equal(payloads[0]["mps_rng"], expected)
    train.restore_rng_state(payloads[0])
    assert len(restored) == 1 and torch.equal(restored[0], expected)


@pytest.mark.parametrize(
    "label", [0.5, 1.2, -0.1, float("nan"), float("inf"), True, "0.5"]
)
def test_prepare_rejects_nonbinary_labels_without_truncation(tmp_path, label):
    from data import prepare_dataset

    rows = [
        {"user": "u", "item": str(index), "label": label if index == 0 else 1}
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="label|binary"):
        prepare_dataset(
            rows, [{"id": str(index)} for index in range(3)], tmp_path / "data.json"
        )


def test_random_cold_catalog_keeps_items_without_interactions(tmp_path):
    import numpy as np

    from data import prepare_dataset

    order = np.random.default_rng(1).permutation(20)
    missing = int(order[2])
    items = [{"id": str(index)} for index in range(20)]
    rows = [
        {"user": "u", "item": str(index)} for index in range(20) if index != missing
    ]
    path = prepare_dataset(rows, items, tmp_path / "data.json", seed=1)
    prepared = json.loads(path.read_text())
    assert prepared["item_pools"]["test"] == sorted(order[2:4].tolist())
    assert sum(map(len, prepared["item_pools"].values())) == 20
    assert prepared["catalog_coverage"]["unassigned_items"] == []


def test_provided_splits_assign_catalog_only_items_only_when_explicit(tmp_path):
    from data import prepare_dataset

    items = [
        {"id": "train"},
        {"id": "val"},
        {"id": "test"},
        {"id": "declared", "split": "test"},
        {"id": "unknown"},
    ]
    rows = [
        {"user": "u", "item": split, "split": split}
        for split in ("train", "val", "test")
    ]
    path = prepare_dataset(rows, items, tmp_path / "data.json")
    prepared = json.loads(path.read_text())
    assert prepared["item_pools"]["test"] == [2, 3]
    assert prepared["catalog_coverage"]["unassigned_items"] == [4]
    assert prepared["catalog_coverage"]["unassigned_item_ids"] == ["unknown"]


def checkpoint_fixture_without_training(
    tmp_path, weight, epoch, best, stale, snapshot=None, capture_best=False
):
    import train

    data_file, features_file = tmp_path / "data.json", tmp_path / "features.npz"
    data_file.write_text("{}")
    features_file.write_bytes(b"software fixture")
    model = SimpleNamespace(
        trainable_state=lambda: {"weight": torch.tensor([weight])},
        llm=SimpleNamespace(metadata=lambda: {"model": "fixture"}),
    )
    optimizer = SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": []})
    return train.save_checkpoint(
        tmp_path / "last.pt",
        model,
        optimizer,
        epoch,
        epoch,
        best,
        stale,
        SimpleNamespace(path=data_file, feature_path=features_file),
        configuration(),
        [],
        best_snapshot=snapshot,
        capture_best=capture_best,
    )


def test_last_checkpoint_snapshot_recovers_interrupted_best_write(tmp_path):
    import train

    first = checkpoint_fixture_without_training(
        tmp_path, 1.0, 1, 0.2, 0, capture_best=True
    )
    train.restore_best_checkpoint(
        tmp_path, torch.load(tmp_path / "last.pt", weights_only=False)
    )
    assert (
        torch.load(tmp_path / "best.pt", weights_only=False)["model"]["weight"].item()
        == 1
    )
    second = checkpoint_fixture_without_training(
        tmp_path, 2.0, 2, 0.3, 0, first, capture_best=True
    )
    assert second["model"]["weight"].item() == 2
    state = torch.load(tmp_path / "last.pt", weights_only=False)
    restored = train.restore_best_checkpoint(tmp_path, state)
    assert restored["best"] == 0.3
    assert (
        torch.load(tmp_path / "best.pt", weights_only=False)["model"]["weight"].item()
        == 2
    )
    checkpoint_fixture_without_training(tmp_path, 3.0, 3, 0.3, 1, restored)
    (tmp_path / "best.pt").unlink()
    train.restore_best_checkpoint(
        tmp_path, torch.load(tmp_path / "last.pt", weights_only=False)
    )
    assert (
        torch.load(tmp_path / "best.pt", weights_only=False)["model"]["weight"].item()
        == 2
    )


def test_best_snapshot_restore_is_atomic_and_legacy_missing_best_is_explicit(
    tmp_path, monkeypatch
):
    import train

    checkpoint_fixture_without_training(tmp_path, 1.0, 1, 0.2, 0, capture_best=True)
    state = torch.load(tmp_path / "last.pt", weights_only=False)
    train.restore_best_checkpoint(tmp_path, state)
    previous = (tmp_path / "best.pt").read_bytes()

    def interrupted(value, path):
        path.write_bytes(b"partial software fixture")
        raise OSError("simulated interruption")

    with monkeypatch.context() as context:
        context.setattr(torch, "save", interrupted)
        with pytest.raises(OSError, match="interruption"):
            train.restore_best_checkpoint(tmp_path, state)
    assert (tmp_path / "best.pt").read_bytes() == previous
    legacy = {key: value for key, value in state.items() if key != "best_snapshot"}
    legacy["stale"] = 1
    (tmp_path / "best.pt").unlink()
    with pytest.raises(ValueError, match="best|snapshot"):
        train.restore_best_checkpoint(tmp_path, legacy)


def test_best_snapshot_has_no_shared_storage_with_live_parameters(tmp_path):
    import train

    data_file, features_file = tmp_path / "data.json", tmp_path / "features.npz"
    data_file.write_text("{}")
    features_file.write_bytes(b"software fixture")
    weight = torch.tensor([1.0])
    model = SimpleNamespace(
        trainable_state=lambda: {"weight": weight},
        llm=SimpleNamespace(metadata=lambda: {"model": "fixture"}),
    )
    optimizer = SimpleNamespace(state_dict=lambda: {"state": {"momentum": weight}})
    snapshot = train.save_checkpoint(
        tmp_path / "last.pt",
        model,
        optimizer,
        1,
        1,
        0.2,
        0,
        SimpleNamespace(path=data_file, feature_path=features_file),
        configuration(),
        [],
        capture_best=True,
    )
    weight.fill_(7.0)
    assert snapshot["model"]["weight"].item() == 1
    assert snapshot["optimizer"]["state"]["momentum"].item() == 1
    train.restore_best_checkpoint(
        tmp_path, torch.load(tmp_path / "last.pt", weights_only=False)
    )
    assert (
        torch.load(tmp_path / "best.pt", weights_only=False)["model"]["weight"].item()
        == 1
    )


@pytest.mark.parametrize(
    "gaps", [(20.0, 30.0), (20.0, 80.0), (-80.0, -20.0), (-120.0, -110.0)]
)
def test_catalog_ranks_stable_log_odds_when_probabilities_saturate(tmp_path, gaps):
    from evaluate import score_catalog

    data_file, features_file = tmp_path / "data.json", tmp_path / "features.npz"
    data_file.write_text("{}")
    features_file.write_bytes(b"software fixture")
    logits = torch.tensor(gaps)
    probabilities = logits.sigmoid()
    dataset = SimpleNamespace(
        eligible_users=lambda split: [0],
        item_pools={"train": [0], "val": [1, 2], "test": [1, 2]},
        positives={"train": {0: {0}}, "val": {0: {2}}, "test": {0: {2}}},
        users=["u"],
        metadata={"protocol": "item_cold"},
        path=data_file,
        feature_path=features_file,
    )
    model = SimpleNamespace(
        config={
            "evaluation": {"ks": [1, 2]},
            "retrieval": {"top_behaviors": 1},
            "ablations": {},
        },
        eval=lambda: None,
        context=lambda dataset, bank, items, user, item, *args: {
            "behaviors": [],
            "scores": torch.empty(0),
            "item": item,
        },
        retriever=SimpleNamespace(rank=lambda *args: []),
        prompt=lambda dataset, items, context, *args: (
            context["item"],
            torch.empty(0, 1),
        ),
        llm=SimpleNamespace(
            score=lambda prompts, *args, **kwargs: {
                "p_yes": probabilities[prompts[0] - 1 : prompts[0]],
                "binary_logit": logits[prompts[0] - 1 : prompts[0]],
            }
        ),
    )
    metrics, rows = score_catalog(
        model, dataset, SimpleNamespace(refresh=lambda: None), torch.zeros(3, 1)
    )
    assert metrics["Recall@1"] == 1
    assert [entry["item"] for entry in rows[0]["ranking"]] == [2, 1]
    assert [entry["ranking_score"] for entry in rows[0]["ranking"]] == [
        gaps[1],
        gaps[0],
    ]
    assert rows[0]["ranking"][0]["score"] == float(probabilities[1])
