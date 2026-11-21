import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from data import RecommendationData, fingerprint, write_json
from evaluate import score_catalog
from model import SuperG
from retrieval import UserBank


def method_recipe(config):
    result = copy.deepcopy(config)
    result.pop("sweeps", None)
    result.pop("features", None)
    result.pop("evaluation", None)
    for key in (
        "device",
        "dtype",
        "local_files_only",
        "feedback_batch_size",
        "gradient_checkpointing",
    ):
        result.get("llm", {}).pop(key, None)
    return result


def completed_training(output, config, data_path, features=None, ablation="full"):
    output = Path(output)
    marker = output / "training_complete.json"
    if not marker.is_file():
        return False
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    saved_config = output / "config.json"
    best = output / "best.pt"
    feature_path = (
        Path(features).resolve()
        if features
        else Path(data_path).resolve().with_name("features.npz")
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("ablation") != ablation
        or not saved_config.is_file()
        or json.loads(saved_config.read_text(encoding="utf-8")) != config
    ):
        raise ValueError("Training completion configuration or ablation mismatch")
    required = {
        "best_checkpoint_sha256": best,
        "dataset_sha256": Path(data_path),
        "features_sha256": feature_path,
        "config_sha256": saved_config,
    }
    for key, path in required.items():
        if not path.is_file() or metadata.get(key) != fingerprint(path):
            raise ValueError(f"Training completion hash mismatch: {key}")
    return True


def restore_rng_state(state):
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if state.get("mps_rng") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_best_checkpoint(output, state):
    path = Path(output) / "best.pt"
    snapshot = state.get("best_snapshot")
    if snapshot is None:
        if state.get("stale") == 0:
            snapshot = {
                key: value for key, value in state.items() if key != "best_snapshot"
            }
        elif path.is_file():
            snapshot = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise ValueError(
                "Checkpoint has no recoverable best snapshot; provide its matching best.pt"
            )
    required = {
        "model",
        "optimizer",
        "epoch",
        "step",
        "best",
        "stale",
        "config",
        "llm",
        "dataset_sha256",
        "feature_sha256",
        "torch_rng",
        "numpy_rng",
        "python_rng",
    }
    if (
        not isinstance(snapshot, dict)
        or not required.issubset(snapshot)
        or snapshot.get("stale") != 0
        or snapshot.get("epoch", -1) > state.get("epoch", -1)
    ):
        raise ValueError("Checkpoint contains an invalid best snapshot")
    for key in ("best", "config", "dataset_sha256", "feature_sha256"):
        if snapshot[key] != state.get(key):
            raise ValueError(f"Best snapshot differs from resume state: {key}")
    if snapshot.get("ablation", "full") != state.get("ablation", "full") or any(
        snapshot["llm"].get(key) != state["llm"].get(key)
        for key in ("model", "resolved_revision")
    ):
        raise ValueError("Best snapshot model or ablation differs from resume state")
    snapshot = _snapshot(
        {key: value for key, value in snapshot.items() if key != "best_snapshot"}
    )
    _atomic_checkpoint(path, snapshot)
    return snapshot


def initialize(config, data_path, features=None, checkpoint=None, ablation=None):
    dataset = RecommendationData(data_path, features)
    if (
        config.get("data", {}).get("protocol", dataset.metadata["protocol"])
        != dataset.metadata["protocol"]
    ):
        raise ValueError("Configured split protocol differs from the prepared data")
    if dataset.features.shape[1] != int(config["dimension"]):
        raise ValueError("Prepared features have the wrong model dimension")
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if method_recipe(state["config"]) != method_recipe(config):
            raise ValueError(
                "Model configuration differs from checkpoint training provenance"
            )
        if ablation is not None and state.get("ablation", "full") != ablation:
            raise ValueError("Requested ablation differs from the trained checkpoint")
        if state["dataset_sha256"] != fingerprint(dataset.path) or state[
            "feature_sha256"
        ] != fingerprint(dataset.feature_path):
            raise ValueError("Checkpoint data or feature contents differ")
        saved_model = state["llm"]
        if saved_model["model"] != config["llm"]["model"]:
            raise ValueError("Checkpoint belongs to another LLM")
        configured = config["llm"].get("revision")
        if configured is not None and configured not in {
            saved_model.get("revision"),
            saved_model.get("resolved_revision"),
        }:
            raise ValueError("LLM revision differs from checkpoint provenance")
    model = SuperG(config)
    if state and state["llm"].get("resolved_revision"):
        model.llm.config["revision"] = state["llm"]["resolved_revision"]
    device = model.load_llm()
    if state:
        resolved = getattr(model.llm.language_model.config, "_commit_hash", None)
        if (
            state["llm"].get("resolved_revision")
            and resolved != state["llm"]["resolved_revision"]
        ):
            raise ValueError("Loaded LLM commit differs from saved provenance")
        model.load_trainable_state(state["model"])
    items = torch.from_numpy(dataset.features).to(device)
    bank = UserBank(
        dataset,
        model.history,
        items,
        int(config["retrieval"].get("bank_batch_size", 256)),
    )
    return model, dataset, items, bank, state


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    step,
    best,
    stale,
    dataset,
    config,
    history,
    ablation="full",
    best_snapshot=None,
    capture_best=False,
):
    state = _snapshot(
        {
            "schema": 1,
            "ablation": ablation,
            "model": model.trainable_state(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best": best,
            "stale": stale,
            "history": history,
            "config": config,
            "llm": model.llm.metadata(),
            "dataset_sha256": fingerprint(dataset.path),
            "feature_sha256": fingerprint(dataset.feature_path),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "mps_rng": torch.mps.get_rng_state()
            if torch.backends.mps.is_available()
            else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
    )
    if capture_best:
        best_snapshot = state.copy()
    if best_snapshot is not None:
        state["best_snapshot"] = best_snapshot
    _atomic_checkpoint(path, state)
    return best_snapshot


def train(config, data_path, output, features=None, resume=None, ablation="full"):
    if ablation not in {"full", "no_suem", "no_ibmm", "no_bhrm"}:
        raise ValueError("Unknown SuperG ablation")
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() != 1
    ):
        raise ValueError("The trainer currently supports one process")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Use a new training output or explicitly resume")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, dataset, items, bank, state = initialize(
        config, data_path, features, resume, ablation
    )
    settings = config["training"]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings.get("learning_rate", 0.001)),
        weight_decay=float(settings.get("weight_decay", 0)),
    )
    epoch, step, best, stale, history = 0, 0, -float("inf"), 0, []
    best_snapshot = None
    if state:
        if state["config"] != config or state.get("ablation", "full") != ablation:
            raise ValueError("Resume configuration or ablation differs")
        best_snapshot = restore_best_checkpoint(output, state)
        optimizer.load_state_dict(state["optimizer"])
        epoch, step, best, stale, history = (
            state["epoch"],
            state["step"],
            state["best"],
            state["stale"],
            state["history"],
        )
        restore_rng_state(state)
    write_json(output / "config.json", config)
    effective_batch = int(settings.get("batch_size", 1024))
    if effective_batch < 1:
        raise ValueError("Training batch size must be positive")
    for current_epoch in range(epoch, int(settings.get("epochs", 100))):
        model.train()
        rng = np.random.default_rng(seed + current_epoch)
        pairs = dataset.sample_training_pairs(
            int(settings.get("negative_ratio", 1)), seed + current_epoch
        )
        if not pairs:
            raise ValueError("No leakage-free training examples remain")
        sums = np.zeros(3)
        count = 0
        candidate_coverage = {
            "candidates": 0,
            "selected": 0,
            "source_users": 0,
            "candidate_shortfall": 0,
            "queries_with_shortfall": 0,
        }
        for offset in range(0, len(pairs), effective_batch):
            batch = pairs[offset : offset + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            if (
                bank.values is None
                or step % int(settings.get("bank_refresh_steps", 1)) == 0
            ):
                bank.refresh()
            contrastive = (
                items.new_zeros(())
                if ablation == "no_suem"
                else model.contrastive(
                    dataset, items, [user for user, item, label in batch], rng
                )
            )
            if contrastive.requires_grad:
                contrastive.backward()
            sum_retrieval = sum_recommendation = 0.0
            for user, item, label in batch:
                retrieval, recommendation, coverage = model.query_loss(
                    dataset, bank, items, user, item, label, ablation
                )
                for key in (
                    "candidates",
                    "selected",
                    "source_users",
                    "candidate_shortfall",
                ):
                    candidate_coverage[key] += coverage[key]
                candidate_coverage["queries_with_shortfall"] += int(
                    coverage["candidate_shortfall"] > 0
                )
                loss = (
                    float(settings.get("alpha", 0.5)) * retrieval
                    + float(settings.get("beta", 0.3)) * recommendation
                ) / len(batch)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite joint objective")
                loss.backward()
                sum_retrieval += float(retrieval.detach())
                sum_recommendation += float(recommendation.detach())
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                float(settings.get("gradient_clip", 1.0)),
            )
            optimizer.step()
            step += 1
            sums += [
                float(contrastive.detach()) * len(batch),
                sum_retrieval,
                sum_recommendation,
            ]
            count += len(batch)
            print(
                json.dumps(
                    {
                        "epoch": current_epoch + 1,
                        "step": step,
                        "examples": count,
                        "contrastive": float(contrastive.detach()),
                        "retrieval": sum_retrieval / len(batch),
                        "recommendation": sum_recommendation / len(batch),
                    }
                ),
                flush=True,
            )
        report = {
            "epoch": current_epoch + 1,
            "step": step,
            "examples": count,
            "candidate_coverage": candidate_coverage,
            "losses": dict(
                zip(
                    ("contrastive", "retrieval", "recommendation"),
                    (sums / count).tolist(),
                )
            ),
            "ablation": ablation,
        }
        if not dataset.eligible_users("val"):
            raise ValueError(
                "Early stopping requires validation users with warm history"
            )
        metrics, _rows = score_catalog(
            model,
            dataset,
            bank,
            items,
            "val",
            ablation,
            config["evaluation"].get("sampled_negatives"),
            seed,
        )
        report["validation"] = metrics
        score = metrics[settings.get("early_stopping_metric", "Recall@20")]
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        history.append(report)
        best_snapshot = save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            current_epoch + 1,
            step,
            best,
            stale,
            dataset,
            config,
            history,
            ablation,
            best_snapshot=best_snapshot,
            capture_best=improved,
        )
        if improved:
            _atomic_checkpoint(output / "best.pt", best_snapshot)
        write_json(output / "history.json", history)
        if stale >= int(settings.get("patience", 5)):
            break
    write_json(
        output / "training_complete.json",
        {
            "ablation": ablation,
            "best_validation_score": best,
            "epochs": len(history),
            "steps": step,
            "best_checkpoint_sha256": fingerprint(output / "best.pt"),
            "dataset_sha256": fingerprint(dataset.path),
            "features_sha256": fingerprint(dataset.feature_path),
            "config_sha256": fingerprint(output / "config.json"),
        },
    )
    return history
