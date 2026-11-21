import importlib
import json
from pathlib import Path
import sys

import numpy as np


def combine_modalities(values, masks=None):
    if not values:
        raise ValueError("At least one modality is required")
    arrays = {name: np.asarray(value, np.float32) for name, value in values.items()}
    shape = next(iter(arrays.values())).shape
    if len(shape) != 2 or any(
        array.shape != shape or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError("Modalities must have equal finite item-by-dimension shape")
    total = np.zeros(shape, np.float32)
    counts = np.zeros((shape[0], 1), np.float32)
    for name, array in arrays.items():
        valid = (
            np.ones(shape[0], bool) if masks is None else np.asarray(masks[name], bool)
        )
        if valid.shape != (shape[0],):
            raise ValueError("Invalid availability mask")
        total += array * valid[:, None]
        counts += valid[:, None]
    if np.any(counts == 0):
        raise ValueError("An item has no available modality")
    return total / counts


def save_features(path, item_ids, modalities, masks=None, metadata=None, dimension=768):
    z = combine_modalities(modalities, masks)
    if z.shape != (len(item_ids), dimension):
        raise ValueError(
            f"Expected {len(item_ids)} item features of dimension {dimension}"
        )
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    availability = {
        f"available_{name}": np.ones(len(item_ids), bool)
        if masks is None
        else np.asarray(masks[name], bool)
        for name in modalities
    }
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            z=z,
            item_ids=np.array(item_ids, str),
            metadata=np.array(json.dumps(metadata or {})),
            **modalities,
            **availability,
        )


def import_features(dataset, files, output, dimension=768, mask_files=None):
    from data import fingerprint

    values = {
        kind: np.load(path, allow_pickle=False) for kind, path in files.items() if path
    }
    mask_files = mask_files or {}
    if any(path and kind not in values for kind, path in mask_files.items()):
        raise ValueError(
            "An availability mask requires the corresponding modality file"
        )
    masks = {
        kind: np.load(mask_files[kind], allow_pickle=False)
        if mask_files.get(kind)
        else np.ones(len(dataset.items), bool)
        for kind in values
    }
    save_features(
        output,
        [str(item["id"]) for item in dataset.items],
        values,
        masks,
        metadata={
            "origin": "provided_features",
            "files": {kind: fingerprint(path) for kind, path in files.items() if path},
            "availability_masks": {
                kind: fingerprint(path) for kind, path in mask_files.items() if path
            },
            "normalization": "none_before_available_modality_mean",
        },
        dimension=dimension,
    )


def load_vision_encoder(name, revision=None, local_files_only=True):
    from transformers import BlipForConditionalGeneration

    full, information = BlipForConditionalGeneration.from_pretrained(
        name,
        revision=revision,
        local_files_only=local_files_only,
        output_loading_info=True,
    )
    if any(key.startswith("vision_model.") for key in information["missing_keys"]):
        raise ValueError(
            "The BLIP checkpoint does not contain all required vision weights"
        )
    vision = full.vision_model
    vision._superg_source_commit = getattr(full.config, "_commit_hash", None)
    return vision


def read_audio(path, max_seconds=None):
    import soundfile as sf
    import torch

    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Audio duration must be finite and positive")
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite nonempty samples")
    waveform = torch.from_numpy(samples.T.copy()).mean(0, keepdim=True)
    if rate != 16000:
        from torchaudio.functional import resample

        waveform = resample(waveform, rate, 16000)
    if max_seconds is not None:
        waveform = waveform[:, : max(1, int(16000 * max_seconds))]
    return waveform


def extract_features(dataset, config, output):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoModel, AutoTokenizer, BlipImageProcessor
    from data import fingerprint

    cfg = config["features"]
    device = torch.device(cfg.get("device", "cpu"))
    local = bool(cfg.get("local_files_only", True))
    revisions = cfg.get("revisions", {})
    values, masks = {}, {}
    resolved = {}
    count = len(dataset.items)
    size = int(config.get("dimension", 768))
    text_indices = [
        i
        for i, item in enumerate(dataset.items)
        if item.get("text") or item.get("title") or item.get("category")
    ]
    if text_indices:
        name = cfg["text_model"]
        tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revisions.get("text"), local_files_only=local
        )
        model = (
            AutoModel.from_pretrained(
                name, revision=revisions.get("text"), local_files_only=local
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        resolved["text"] = getattr(model.config, "_commit_hash", None)
        values["text"] = np.zeros((count, size), np.float32)
        masks["text"] = np.zeros(count, bool)
        for start in range(0, len(text_indices), int(cfg.get("batch_size", 16))):
            ids = text_indices[start : start + int(cfg.get("batch_size", 16))]
            texts = [
                " ".join(
                    str(dataset.items[i].get(key, ""))
                    for key in ("title", "category", "text")
                )
                for i in ids
            ]
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=int(cfg.get("max_text_length", 512)),
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output_values = (
                    model(**tokens).last_hidden_state[:, 0].float().cpu().numpy()
                )
            values["text"][ids] = output_values
            masks["text"][ids] = True
        del model
    image_indices = [i for i, item in enumerate(dataset.items) if item.get("image")]
    if image_indices:
        name = cfg["vision_model"]
        processor = BlipImageProcessor.from_pretrained(
            name, revision=revisions.get("vision"), local_files_only=local
        )
        model = (
            load_vision_encoder(name, revisions.get("vision"), local)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        resolved["vision"] = model._superg_source_commit
        values["visual"] = np.zeros((count, size), np.float32)
        masks["visual"] = np.zeros(count, bool)
        for start in range(0, len(image_indices), int(cfg.get("batch_size", 16))):
            ids = image_indices[start : start + int(cfg.get("batch_size", 16))]
            images = []
            for i in ids:
                with Image.open(dataset.items[i]["image"]) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            batch = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                output_values = model(**batch).pooler_output.float().cpu().numpy()
            values["visual"][ids] = output_values
            masks["visual"][ids] = True
        del model
    audio_indices = [i for i, item in enumerate(dataset.items) if item.get("audio")]
    if audio_indices:
        source = Path(cfg["beats_source"]).resolve()
        if not (source / "BEATs.py").is_file():
            raise FileNotFoundError(
                "features.beats_source must point to the official local BEATs source"
            )
        sys.path.insert(0, str(source))
        try:
            module = importlib.import_module("BEATs")
        finally:
            sys.path.pop(0)
        checkpoint = torch.load(
            cfg["beats_checkpoint"], map_location="cpu", weights_only=False
        )
        resolved["beats_checkpoint_sha256"] = fingerprint(cfg["beats_checkpoint"])
        beats_cfg = module.BEATsConfig(checkpoint["cfg"])
        beats_cfg.finetuned_model = False
        model = module.BEATs(beats_cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval().requires_grad_(False)
        values["audio"] = np.zeros((count, size), np.float32)
        masks["audio"] = np.zeros(count, bool)
        for i in audio_indices:
            waveform = read_audio(
                dataset.items[i]["audio"], cfg.get("max_audio_seconds")
            )
            with torch.no_grad():
                frames, mask = model.extract_features(
                    waveform.to(device),
                    padding_mask=torch.zeros_like(
                        waveform, dtype=torch.bool, device=device
                    ),
                )
                valid = (
                    torch.ones(frames.shape[:2], dtype=torch.bool, device=device)
                    if mask is None
                    else ~mask
                )
                embedding = (frames * valid.unsqueeze(-1)).sum(1) / valid.sum(
                    1, keepdim=True
                ).clamp_min(1)
            values["audio"][i] = embedding[0].float().cpu().numpy()
            masks["audio"][i] = True
        del model
    save_features(
        output,
        [str(item["id"]) for item in dataset.items],
        values,
        masks,
        {
            "origin": "raw_extraction",
            "configuration": cfg,
            "resolved_revisions": resolved,
            "normalization": "none_before_available_modality_mean",
        },
        dimension=size,
    )
