import json
from pathlib import Path

import numpy as np
import torch

from data import fingerprint, write_json


def ranking_metrics(rankings, positives, ks=(10, 20)):
    if set(rankings) != set(positives) or not positives:
        raise ValueError("Rankings must cover exactly the nonempty evaluated user set")
    result = {f"{metric}@{k}": [] for metric in ("Recall", "NDCG") for k in ks}
    for user, truth in positives.items():
        truth = set(truth)
        ranked = rankings[user]
        if not truth or len(ranked) != len(set(ranked)):
            raise ValueError("Empty ground truth or duplicate ranked items")
        for k in ks:
            if k < 1:
                raise ValueError("Ranking cutoffs must be positive")
            hits = np.array([item in truth for item in ranked[:k]], np.float64)
            result[f"Recall@{k}"].append(float(hits.sum() / len(truth)))
            discount = 1 / np.log2(np.arange(len(hits)) + 2)
            ideal = (1 / np.log2(np.arange(min(k, len(truth))) + 2)).sum()
            result[f"NDCG@{k}"].append(float((hits * discount).sum() / ideal))
    return {
        **{key: float(np.mean(values)) for key, values in result.items()},
        "users": len(positives),
    }


@torch.no_grad()
def score_catalog(
    model,
    dataset,
    bank,
    items,
    split="test",
    ablation="full",
    sampled_negatives=None,
    seed=1,
):
    if split not in {"val", "test"}:
        raise ValueError("Catalog evaluation is restricted to validation or test")
    model.eval()
    bank.refresh()
    eligible = dataset.eligible_users(split)
    rng = np.random.default_rng(seed)
    pool = sorted(dataset.item_pools[split])
    rankings, truth, rows = {}, {}, []
    max_k = max(model.config["evaluation"].get("ks", [10, 20]))
    for user in eligible:
        positive = dataset.positives[split][user]
        candidates = [
            item
            for item in pool
            if item not in dataset.positives["train"].get(user, set())
        ]
        if not positive.issubset(candidates):
            raise ValueError("Evaluation candidate pool excludes ground truth")
        if sampled_negatives is not None:
            negative = [item for item in candidates if item not in positive]
            chosen = rng.choice(
                negative, min(int(sampled_negatives), len(negative)), replace=False
            ).tolist()
            candidates = sorted(set(chosen) | positive)
        scored = []
        adaptation = (
            model.config["ablations"].get("no_bhrm_adaptation", "zero_pad")
            if ablation == "no_bhrm"
            else "learned"
        )
        for item in candidates:
            context = model.context(dataset, bank, items, user, item, False, ablation)
            behaviors = model.retriever.rank(
                context["behaviors"],
                context["scores"],
                int(model.config["retrieval"]["top_behaviors"]),
            )
            prompt, slots = model.prompt(dataset, items, context, behaviors, adaptation)
            output = model.llm.score([prompt], [slots], adaptation=adaptation)
            probability = float(output["p_yes"][0].cpu())
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    f"Invalid p_yes probability for user {user}, item {item}: {probability}"
                )
            ranking_score = float(output["binary_logit"][0].cpu())
            if not np.isfinite(ranking_score):
                raise ValueError(
                    f"Invalid binary ranking score for user {user}, item {item}: {ranking_score}"
                )
            scored.append(
                {
                    "item": item,
                    "score": probability,
                    "ranking_score": ranking_score,
                    "candidate_behaviors": len(context["behaviors"]),
                    "behaviors": [
                        {
                            "user": behavior.user,
                            "item": behavior.item,
                            "label": behavior.label,
                        }
                        for behavior in behaviors
                    ],
                }
            )
        scored.sort(key=lambda value: (-value["ranking_score"], value["item"]))
        rankings[user] = [value["item"] for value in scored[:max_k]]
        truth[user] = positive
        rows.append(
            {
                "user": user,
                "user_id": dataset.users[user],
                "candidate_count": len(candidates),
                "candidate_ids": candidates,
                "positive_items": sorted(positive),
                "ranking": scored[:max_k],
            }
        )
    metrics = ranking_metrics(
        rankings, truth, model.config["evaluation"].get("ks", [10, 20])
    )
    metrics.update(
        {
            "split": split,
            "protocol": dataset.metadata["protocol"],
            "candidate_mode": "full_split_catalog"
            if sampled_negatives is None
            else "sampled_negatives",
            "sampled_negatives": sampled_negatives,
            "available_users": len(dataset.positives[split]),
            "excluded_without_warm_history": len(dataset.positives[split])
            - len(eligible),
            "candidate_items": len(pool),
            "ablation": ablation,
            "dataset_sha256": fingerprint(dataset.path),
            "feature_sha256": fingerprint(dataset.feature_path),
        }
    )
    return metrics, rows


def save_evaluation(output, metrics, rows, config, checkpoint):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rankings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(
        output / "metrics.json",
        {**metrics, "config": config, "checkpoint_sha256": fingerprint(checkpoint)},
    )


def aggregate(paths, output):
    from train import method_recipe

    reports = [json.loads(Path(path).read_text()) for path in paths]
    groups = {}
    for report in reports:
        cfg = report["config"]
        recipe = method_recipe(cfg)
        recipe.get("training", {}).pop("seed", None)
        recipe.get("training", {}).pop("output", None)
        recipe["evaluation"] = cfg.get("evaluation", {})
        key = json.dumps(
            {
                "dataset": report["dataset_sha256"],
                "features": report["feature_sha256"],
                "ablation": report["ablation"],
                "protocol": report["protocol"],
                "candidate_mode": report["candidate_mode"],
                "sampled_negatives": report["sampled_negatives"],
                "split": report["split"],
                "method": recipe,
            },
            sort_keys=True,
        )
        groups.setdefault(key, []).append(report)
    rows = []
    for key, values in groups.items():
        seeds = sorted(value["config"]["training"]["seed"] for value in values)
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seed within an identical evaluation protocol")
        names = [name for name in values[0] if name.startswith(("Recall@", "NDCG@"))]
        if not names or any(
            {name for name in value if name.startswith(("Recall@", "NDCG@"))}
            != set(names)
            for value in values
        ):
            raise ValueError("Aggregation requires identical measured ranking metrics")
        if any(
            isinstance(value[name], bool)
            or not isinstance(value[name], (int, float))
            or not np.isfinite(value[name])
            or not 0 <= value[name] <= 1
            for value in values
            for name in names
        ):
            raise ValueError(
                "Measured ranking metrics must be finite fractions in [0,1]"
            )
        rows.append(
            {
                "protocol": json.loads(key),
                "seeds": seeds,
                "five_seed_complete": len(seeds) == 5,
                "metrics": {
                    name: {
                        "mean": float(np.mean([value[name] for value in values])),
                        "std": float(np.std([value[name] for value in values], ddof=1))
                        if len(values) > 1
                        else 0.0,
                    }
                    for name in names
                },
            }
        )
    write_json(
        output,
        {"measured_runs": len(reports), "groups": rows, "paper_values_included": False},
    )
    return rows
