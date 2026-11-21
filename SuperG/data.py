import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path):
    path = Path(path)
    if path.suffix == ".json":
        result = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(result, dict):
            return [{"id": key, **value} for key, value in result.items()]
        return result
    if path.suffix in (".tsv", ".csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            return list(
                csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ",")
            )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _binary_label(value):
    if isinstance(value, str):
        if value.strip() not in {"0", "1"}:
            raise ValueError("Interaction labels must be binary 0 or 1")
        return int(value.strip())
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value not in (0, 1)
    ):
        raise ValueError("Interaction labels must be binary 0 or 1")
    return int(value)


def prepare_dataset(
    interactions, items, output, seed=1, protocol="item_cold", name="custom"
):
    if protocol not in {"item_cold", "interaction_random"}:
        raise ValueError("Unknown splitting protocol")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = (
        read_records(interactions)
        if isinstance(interactions, (str, Path))
        else list(interactions)
    )
    metadata = read_records(items) if isinstance(items, (str, Path)) else list(items)
    base = (
        Path(items).resolve().parent if isinstance(items, (str, Path)) else Path.cwd()
    )
    item_ids = [str(item["id"]) for item in metadata]
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("Item IDs must be nonempty and unique")
    for item in metadata:
        item["id"] = str(item["id"])
        for key in ("image", "audio"):
            if item.get(key):
                path = Path(item[key]).expanduser()
                item[key] = str((path if path.is_absolute() else base / path).resolve())
    user_ids = sorted({str(row["user"]) for row in rows})
    if not user_ids:
        raise ValueError("The interaction dataset is empty")
    user_map = {user: index for index, user in enumerate(user_ids)}
    item_map = {item: index for index, item in enumerate(item_ids)}
    rng = np.random.default_rng(seed)
    item_order = rng.permutation(len(item_ids))
    if len(item_ids) < 3:
        raise ValueError("At least three items are required for three disjoint splits")
    n_val = max(1, int(len(item_ids) * 0.1))
    n_test = max(1, int(len(item_ids) * 0.1))
    pools = {
        "val": set(item_order[:n_val].tolist()),
        "test": set(item_order[n_val : n_val + n_test].tolist()),
        "train": set(item_order[n_val + n_test :].tolist()),
    }
    item_split = {item: split for split, ids in pools.items() for item in ids}
    provided = all(row.get("split") in {"train", "val", "test"} for row in rows)
    if any(row.get("split") for row in rows) and not provided:
        raise ValueError("Supplied split labels must exist on every interaction")
    interaction_order = rng.permutation(len(rows))
    interaction_split = {
        int(index): "train"
        if rank < int(0.8 * len(rows))
        else "val"
        if rank < int(0.9 * len(rows))
        else "test"
        for rank, index in enumerate(interaction_order)
    }
    normalized = []
    seen = set()
    for position, row in enumerate(rows):
        user, item = str(row["user"]), str(row["item"])
        if item not in item_map:
            raise ValueError(f"Interaction references an unknown item: {item}")
        label = _binary_label(row.get("label", 1))
        pair = (user, item)
        if pair in seen:
            raise ValueError(
                f"Duplicate user/item pair; deduplicate explicitly first: {pair}"
            )
        seen.add(pair)
        split = (
            row["split"]
            if provided
            else item_split[item_map[item]]
            if protocol == "item_cold"
            else interaction_split[position]
        )
        timestamp = (
            float(row["timestamp"])
            if row.get("timestamp") not in (None, "")
            else float(position)
        )
        normalized.append(
            {
                "user": user_map[user],
                "item": item_map[item],
                "label": label,
                "timestamp": timestamp,
                "position": position,
                "split": split,
            }
        )
    actual_pools = {
        split: {row["item"] for row in normalized if row["split"] == split}
        for split in ("train", "val", "test")
    }
    if protocol == "item_cold" and not provided:
        actual_pools = pools
    elif provided:
        for index, item in enumerate(metadata):
            declared = item.get("split")
            if declared not in (None, "", "train", "val", "test"):
                raise ValueError("Invalid explicit item split")
            memberships = {
                split for split, pool in actual_pools.items() if index in pool
            }
            if declared in {"train", "val", "test"}:
                if (
                    protocol == "item_cold"
                    and memberships
                    and memberships != {declared}
                ):
                    raise ValueError(
                        "Item metadata split conflicts with interaction splits"
                    )
                if not memberships:
                    actual_pools[declared].add(index)
    actual_pools = {split: sorted(pool) for split, pool in actual_pools.items()}
    assigned = set().union(*actual_pools.values())
    unassigned = sorted(set(range(len(metadata))) - assigned)
    if protocol == "item_cold" and any(
        set(actual_pools[a]) & set(actual_pools[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Provided splits violate strict item cold-start separation")
    payload = {
        "schema": 1,
        "name": name,
        "protocol": protocol,
        "seed": int(seed),
        "users": user_ids,
        "items": metadata,
        "interactions": normalized,
        "item_pools": actual_pools,
        "history_order": "timestamp_then_input_position",
        "split_origin": "provided" if provided else "seeded_random",
        "catalog_coverage": {
            "total_items": len(metadata),
            "assigned_items": len(assigned),
            "unassigned_items": unassigned,
            "unassigned_item_ids": [item_ids[index] for index in unassigned],
            "pool_basis": "all_metadata_seeded_item_split"
            if protocol == "item_cold" and not provided
            else "provided_interactions_and_explicit_item_splits"
            if provided
            else "observed_interaction_splits",
        },
    }
    write_json(output, payload)
    return output


def import_milk(root, output, seed=1, protocol="item_cold"):
    root = Path(root)
    rows = []
    maximum = -1
    for split in ("train", "val", "test"):
        for line in (root / f"{split}.txt").read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            for index, item in enumerate(fields[1:]):
                maximum = max(maximum, int(item))
                rows.append(
                    {
                        "user": fields[0],
                        "item": item,
                        "timestamp": index,
                        "split": split,
                    }
                )
    return prepare_dataset(
        rows,
        [{"id": str(index), "text": ""} for index in range(maximum + 1)],
        output,
        seed,
        protocol,
        root.name,
    )


class RecommendationData:
    def __init__(self, path, features=None, require_features=True):
        self.path = Path(path).resolve()
        self.metadata = json.loads(self.path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != 1:
            raise ValueError("Unsupported dataset schema")
        self.users = self.metadata["users"]
        self.items = self.metadata["items"]
        self.rows = self.metadata["interactions"]
        self.item_pools = self.metadata["item_pools"]
        if len(self.users) != len(set(self.users)) or len(self.items) != len(
            {str(item["id"]) for item in self.items}
        ):
            raise ValueError("Prepared entity IDs are not unique")
        if set(self.item_pools) != {"train", "val", "test"}:
            raise ValueError("Prepared data needs all three item pools")
        for pool in self.item_pools.values():
            if len(pool) != len(set(pool)) or any(
                not isinstance(item, int) or not 0 <= item < len(self.items)
                for item in pool
            ):
                raise ValueError("Invalid prepared item pool")
        seen = set()
        for row in self.rows:
            split = row.get("split")
            if (
                split not in self.item_pools
                or row.get("item") not in self.item_pools[split]
            ):
                raise ValueError(
                    "Interaction split differs from its declared item pool"
                )
            if (
                type(row.get("label")) is not int
                or row["label"] not in (0, 1)
                or not np.isfinite(float(row["timestamp"]))
            ):
                raise ValueError("Invalid prepared interaction label or timestamp")
            key = (row["user"], row["item"])
            if key in seen:
                raise ValueError("Duplicate prepared user/item pair")
            seen.add(key)
        self.histories = [[] for _ in self.users]
        self.behavior_histories = [[] for _ in self.users]
        self.behavior_labels = {}
        self.item_users = [[] for _ in self.items]
        self.positives = {split: {} for split in ("train", "val", "test")}
        for row in sorted(
            self.rows, key=lambda row: (row["timestamp"], row["position"])
        ):
            u, j = row["user"], row["item"]
            if not 0 <= u < len(self.users) or not 0 <= j < len(self.items):
                raise ValueError("Invalid entity index in prepared data")
            if row["split"] == "train":
                self.behavior_histories[u].append(j)
                self.behavior_labels[(u, j)] = row["label"]
            if row["label"] == 1:
                self.positives[row["split"]].setdefault(u, set()).add(j)
                if row["split"] == "train":
                    self.histories[u].append(j)
                    self.item_users[j].append(u)
        self.feature_path = (
            Path(features).resolve()
            if features
            else self.path.with_name("features.npz")
        )
        self.features = None
        self.feature_metadata = {}
        if require_features or self.feature_path.exists():
            with np.load(self.feature_path, allow_pickle=False) as archive:
                if list(archive["item_ids"].astype(str)) != [
                    str(item["id"]) for item in self.items
                ]:
                    raise ValueError(
                        "Feature row IDs differ from the prepared item order"
                    )
                self.features = np.asarray(archive["z"], np.float32)
                self.feature_metadata = json.loads(str(archive["metadata"]))
            if (
                self.features.ndim != 2
                or len(self.features) != len(self.items)
                or not np.isfinite(self.features).all()
            ):
                raise ValueError("Invalid item feature matrix")
        if self.metadata["protocol"] == "item_cold":
            groups = [set(self.item_pools[s]) for s in ("train", "val", "test")]
            if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
                raise ValueError("Cold item pools overlap")

    def history(self, user, exclude_item=None, max_length=None):
        result = [item for item in self.histories[user] if item != exclude_item]
        return result[-max_length:] if max_length else result

    def eligible_users(self, split):
        return sorted(user for user in self.positives[split] if self.histories[user])

    def sample_training_pairs(self, negative_ratio=1, seed=1):
        rng = np.random.default_rng(seed)
        pool = np.array(self.item_pools["train"], dtype=np.int64)
        result = []
        for row in self.rows:
            if row["split"] != "train":
                continue
            u, j, label = row["user"], row["item"], row["label"]
            if not self.history(u, exclude_item=j):
                continue
            result.append((u, j, label))
            if label == 1:
                negative_pool = np.array(
                    [
                        item
                        for item in pool
                        if item not in self.positives["train"].get(u, set())
                    ]
                )
                if negative_ratio and not len(negative_pool):
                    continue
                for item in (
                    rng.choice(negative_pool, size=int(negative_ratio), replace=True)
                    if negative_ratio
                    else []
                ):
                    result.append((u, int(item), 0))
        rng.shuffle(result)
        return result
