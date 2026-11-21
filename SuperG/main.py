import argparse
import copy
import glob
import json
import subprocess
import sys
from pathlib import Path

import yaml

from data import RecommendationData, import_milk, prepare_dataset, write_json


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Configuration overrides require dotted.key=value")
        target = config
        fields = key.split(".")
        for field in fields[:-1]:
            target = target.setdefault(field, {})
        target[fields[-1]] = yaml.safe_load(value)
    return config


def parser():
    cli = argparse.ArgumentParser(
        description="SuperG multimodal cold-item recommendation"
    )
    cli.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    cli.add_argument("--set", action="append", default=[])
    commands = cli.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--interactions")
    prepare.add_argument("--items")
    prepare.add_argument("--milk-root")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--name", default="custom")
    features = commands.add_parser("features")
    features.add_argument("--data", required=True)
    features.add_argument("--output", required=True)
    for name in ("visual", "text", "audio"):
        features.add_argument(f"--{name}-npy")
        features.add_argument(f"--{name}-mask-npy")
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--features")
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument(
        "--ablation", choices=("full", "no_suem", "no_ibmm", "no_bhrm"), default="full"
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--features")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument(
        "--ablation", choices=("full", "no_suem", "no_ibmm", "no_bhrm"), default="full"
    )
    sweep = commands.add_parser("sweep")
    sweep.add_argument("--data", required=True)
    sweep.add_argument("--features")
    sweep.add_argument("--output", required=True)
    sweep.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    sweep.add_argument("--execute", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--input", nargs="+", required=True)
    report.add_argument("--output", required=True)
    commands.add_parser("protocol")
    return cli


def sweep_jobs(args, config):
    from train import completed_training

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = [("full", "full", {})] + [
        (name, name, {}) for name in ("no_suem", "no_ibmm", "no_bhrm")
    ]
    scans = config.get("sweeps", {})
    for key, values in scans.items():
        for value in values:
            settings.append(
                (key.replace(".", "_") + "_" + str(value), "full", {key: value})
            )
    jobs = []
    recipes = set()
    for label, ablation, changes in settings:
        for seed in args.seeds:
            cfg = copy.deepcopy(config)
            cfg["training"]["seed"] = seed
            for key, value in changes.items():
                fields = key.split(".")
                target = cfg
                for field in fields[:-1]:
                    target = target[field]
                target[fields[-1]] = value
            identity = json.dumps({"config": cfg, "ablation": ablation}, sort_keys=True)
            if identity in recipes:
                continue
            recipes.add(identity)
            folder = output / label / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            checkpoints = folder / "checkpoints"
            complete = completed_training(
                checkpoints, cfg, args.data, args.features, ablation
            )
            config_path = folder / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )
            common = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(config_path),
            ]
            train = common + [
                "train",
                "--data",
                str(Path(args.data).resolve()),
                "--output",
                str(folder / "checkpoints"),
                "--ablation",
                ablation,
            ]
            evaluate = common + [
                "evaluate",
                "--data",
                str(Path(args.data).resolve()),
                "--checkpoint",
                str(folder / "checkpoints" / "best.pt"),
                "--output",
                str(folder / "test"),
                "--ablation",
                ablation,
            ]
            if args.features:
                train.extend(["--features", str(Path(args.features).resolve())])
                evaluate.extend(["--features", str(Path(args.features).resolve())])
            if not complete and (checkpoints / "last.pt").is_file():
                train.extend(["--resume", str(checkpoints / "last.pt")])
            jobs.append(
                {
                    "label": label,
                    "seed": seed,
                    "ablation": ablation,
                    "train": train,
                    "evaluate": evaluate,
                    "training_complete": complete,
                }
            )
    write_json(output / "jobs.json", jobs)
    if args.execute:
        for job in jobs:
            if not job["training_complete"]:
                subprocess.run(job["train"], check=True)
            subprocess.run(job["evaluate"], check=True)
    return jobs


def main():
    args = parser().parse_args()
    config = load_config(args.config, args.set)
    if args.command == "prepare":
        if args.milk_root:
            result = import_milk(
                args.milk_root,
                args.output,
                config["data"]["seed"],
                config["data"]["protocol"],
            )
        else:
            if not args.interactions or not args.items:
                raise ValueError(
                    "prepare requires --interactions and --items, or --milk-root"
                )
            result = prepare_dataset(
                args.interactions,
                args.items,
                args.output,
                config["data"]["seed"],
                config["data"]["protocol"],
                args.name,
            )
        print(result)
    elif args.command == "features":
        from features import extract_features, import_features

        dataset = RecommendationData(args.data, require_features=False)
        files = {
            kind: getattr(args, f"{kind}_npy") for kind in ("visual", "text", "audio")
        }
        masks = {
            kind: getattr(args, f"{kind}_mask_npy")
            for kind in ("visual", "text", "audio")
        }
        if any(files.values()):
            import_features(dataset, files, args.output, config["dimension"], masks)
        else:
            if any(masks.values()):
                raise ValueError(
                    "Availability masks require imported modality features"
                )
            extract_features(dataset, config, args.output)
    elif args.command == "train":
        from train import train

        train(config, args.data, args.output, args.features, args.resume, args.ablation)
    elif args.command == "evaluate":
        from evaluate import save_evaluation, score_catalog
        from train import initialize

        model, dataset, items, bank, _state = initialize(
            config, args.data, args.features, args.checkpoint, args.ablation
        )
        metrics, rows = score_catalog(
            model,
            dataset,
            bank,
            items,
            args.split,
            args.ablation,
            config["evaluation"].get("sampled_negatives"),
            config["training"]["seed"],
        )
        save_evaluation(args.output, metrics, rows, config, args.checkpoint)
        print(json.dumps(metrics))
    elif args.command == "sweep":
        jobs = sweep_jobs(args, config)
        print(json.dumps({"jobs": len(jobs), "executed": args.execute}))
    elif args.command == "report":
        from evaluate import aggregate

        paths = sorted(
            {
                path
                for expression in args.input
                for path in glob.glob(expression, recursive=True)
            }
        )
        if not paths:
            raise FileNotFoundError("No measured metric files matched")
        aggregate(paths, args.output)
    else:
        print(
            json.dumps(
                {
                    "model": "SuperG",
                    "configuration": config,
                    "original_conflicts": [
                        "Eq2 denominator excludes positives",
                        "N=5 implementation versus N=10 sensitivity",
                        "item split versus interaction split",
                        "prompt instruction first versus last",
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
