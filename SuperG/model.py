import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from retrieval import Behavior, BehaviorRetriever


def augment_history(history, method, rng, ratio=0.2, segments=3):
    history = list(history)
    if not history:
        raise ValueError("Cannot augment an empty history")
    if method not in {"mask", "sample", "reorder", "segments"} or not 0 < ratio <= 1:
        raise ValueError("Invalid history augmentation")
    size = max(1, min(len(history), int(np.ceil(len(history) * ratio))))
    if method == "mask":
        keep = rng.choice(len(history), max(1, len(history) - size), replace=False)
        return [history[i] for i in sorted(keep)]
    start = int(rng.integers(0, len(history) - size + 1))
    if method == "sample":
        return history[start : start + size]
    if method == "reorder":
        selected = history[start : start + size]
        rng.shuffle(selected)
        return history[:start] + selected + history[start + size :]
    pieces = [
        part.tolist()
        for part in np.array_split(
            np.asarray(history), min(len(history), max(1, segments))
        )
    ]
    return [item for index in rng.permutation(len(pieces)) for item in pieces[index]]


def contrastive_loss(first, second, temperature=0.1, denominator="negative_only"):
    if (
        first.shape != second.shape
        or first.ndim != 2
        or first.shape[0] < 2
        or temperature <= 0
    ):
        raise ValueError(
            "Contrastive learning requires at least two distinct users with matching views"
        )
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / temperature
    positive = logits.diag()
    if denominator == "negative_only":
        masked = logits.masked_fill(
            torch.eye(len(first), dtype=torch.bool, device=first.device), -torch.inf
        )
        return (masked.logsumexp(1) + masked.logsumexp(0) - 2 * positive).mean()
    if denominator == "include_positive":
        return (logits.logsumexp(1) + logits.logsumexp(0) - 2 * positive).mean()
    raise ValueError("Unknown contrastive denominator")


def retrieval_kl(scores, feedback):
    if scores.shape != feedback.shape or scores.ndim != 1 or not len(scores):
        raise ValueError("KL inputs must be equal nonempty candidate vectors")
    log_r = scores.log_softmax(0)
    log_l = feedback.log_softmax(0)
    return (log_r.exp() * (log_r - log_l)).sum()


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        dimension=768,
        heads=2,
        layers=1,
        hidden=3072,
        max_history=100,
        dropout=0.1,
    ):
        super().__init__()
        self.max_history = int(max_history)
        self.positions = nn.Embedding(self.max_history, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension, heads, hidden, dropout, batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False
        )

    def encode(self, histories, item_features, positions=None):
        if not histories or any(not history for history in histories):
            raise ValueError("History encoder requires nonempty user histories")
        histories = [list(history)[-self.max_history :] for history in histories]
        size = max(map(len, histories))
        indices = torch.zeros(
            (len(histories), size), dtype=torch.long, device=item_features.device
        )
        valid = torch.zeros_like(indices, dtype=torch.bool)
        locations = torch.zeros_like(indices)
        for row, history in enumerate(histories):
            indices[row, : len(history)] = torch.tensor(history, device=indices.device)
            valid[row, : len(history)] = True
            location = (
                list(range(len(history)))
                if positions is None
                else list(positions[row])[-self.max_history :]
            )
            if len(location) != len(history) or any(
                value < 0 or value >= self.max_history for value in location
            ):
                raise ValueError("Invalid explicit positional indices")
            locations[row, : len(history)] = torch.tensor(
                location, device=indices.device
            )
        values = item_features[indices] + self.positions(locations)
        values = self.transformer(values, src_key_padding_mask=~valid)
        return (values * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)


class FrozenBinaryLLM(nn.Module):
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__()
        self.config = dict(config)
        self.language_model = model
        self.tokenizer = tokenizer
        self.adapter = None
        self.label_ids = None
        if model is not None:
            self._configure()

    def _configure(self):
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("A native chat template is required")
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[USER_EMBED]", "[ITEM_EMBED]"]}
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if (
            len(self.tokenizer)
            != self.language_model.get_input_embeddings().num_embeddings
        ):
            self.language_model.resize_token_embeddings(
                len(self.tokenizer), mean_resizing=False
            )
        self.tokenizer.padding_side = "left"
        self.language_model.requires_grad_(False).eval()
        hidden = self.language_model.config.hidden_size
        dimension = int(self.config.get("dimension", 768))
        bottleneck = int(self.config.get("adapter_hidden", 256))
        self.adapter = nn.Sequential(
            nn.Linear(dimension, bottleneck), nn.ReLU(), nn.Linear(bottleneck, hidden)
        )
        device = next(self.language_model.parameters()).device
        self.adapter.to(device)
        ids = [
            self.tokenizer.encode(word, add_special_tokens=False)
            for word in ("No", "Yes")
        ]
        if any(len(value) != 1 for value in ids) or ids[0] == ids[1]:
            raise ValueError("Yes and No must be distinct single-token responses")
        self.label_ids = [value[0] for value in ids]
        if self.config.get("gradient_checkpointing", False):
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def load(self):
        if self.language_model is not None:
            return self
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.config.get("device", "auto")
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        name = self.config.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
        revision = self.config.get("revision")
        local = bool(self.config.get("local_files_only", True))
        dtype = self.config.get("dtype", "auto")
        if dtype == "auto":
            dtype = (
                "float32"
                if device == "cpu"
                else "bfloat16"
                if device.startswith("cuda")
                else "float16"
            )
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Invalid LLM dtype")
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revision, local_files_only=local, trust_remote_code=False
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            local_files_only=local,
            trust_remote_code=False,
            dtype=getattr(torch, dtype),
        ).to(device)
        self._configure()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.language_model is not None:
            self.language_model.eval()
        return self

    def score(self, prompts, slots, distribution=None, adaptation="learned"):
        self.load()
        if len(prompts) != len(slots) or not prompts:
            raise ValueError(
                "Prompts and entity slots must form a nonempty aligned batch"
            )
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        for text in rendered:
            prefix = self.tokenizer.encode(text, add_special_tokens=False)
            for label, label_id in zip(("No", "Yes"), self.label_ids):
                continuation = self.tokenizer.encode(
                    text + label, add_special_tokens=False
                )
                if continuation != prefix + [label_id]:
                    raise ValueError(
                        "The native chat boundary does not yield one Yes/No token"
                    )
        tokens = self.tokenizer(
            rendered,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
            return_token_type_ids=False,
        )
        limit = min(
            int(self.config.get("max_length", 2048)),
            int(self.language_model.config.max_position_embeddings),
        )
        if tokens["input_ids"].shape[1] + 1 > limit:
            raise ValueError(
                f"Hybrid prompt exceeds the {limit}-token context; truncation is disabled"
            )
        device = next(self.language_model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        embeds = (
            self.language_model.get_input_embeddings()(tokens["input_ids"])
            .detach()
            .clone()
        )
        user_id = self.tokenizer.convert_tokens_to_ids("[USER_EMBED]")
        item_id = self.tokenizer.convert_tokens_to_ids("[ITEM_EMBED]")
        for row, entities in enumerate(slots):
            locations = (
                (tokens["input_ids"][row] == user_id)
                | (tokens["input_ids"][row] == item_id)
            ).nonzero(as_tuple=True)[0]
            if len(locations) != len(entities):
                raise ValueError(
                    "Entity slot count differs from USER/ITEM marker occurrences"
                )
            entities = entities.to(device=device, dtype=self.adapter[0].weight.dtype)
            if adaptation == "learned":
                converted = self.adapter(entities)
            elif adaptation == "zero_pad":
                if entities.shape[-1] > embeds.shape[-1]:
                    raise ValueError(
                        "Direct embeddings are wider than the LLM token space"
                    )
                converted = F.pad(entities, (0, embeds.shape[-1] - entities.shape[-1]))
            elif adaptation == "text":
                if len(entities):
                    raise ValueError(
                        "Numeric-text prompts must not contain embedding markers"
                    )
                continue
            else:
                raise ValueError("Unknown embedding adaptation")
            embeds[row, locations] = converted.to(embeds.dtype)

        def final_logits(values, mask):
            return (
                self.language_model(
                    inputs_embeds=values, attention_mask=mask, use_cache=False
                )
                .logits[:, -1]
                .float()
            )

        if (
            self.config.get("gradient_checkpointing", False)
            and torch.is_grad_enabled()
            and embeds.requires_grad
        ):
            from torch.utils.checkpoint import checkpoint

            logits = checkpoint(
                final_logits, embeds, tokens["attention_mask"], use_reentrant=False
            )
        else:
            logits = final_logits(embeds, tokens["attention_mask"])
        selected = logits[:, self.label_ids]
        mode = distribution or self.config.get("prediction_distribution", "yes_no")
        if mode == "full_vocab":
            log_probabilities = logits.log_softmax(-1)[:, self.label_ids]
            complementary = logits.clone()
            complementary[:, self.label_ids[1]] = -torch.inf
            binary_logit = logits[:, self.label_ids[1]] - complementary.logsumexp(-1)
        elif mode == "yes_no":
            log_probabilities = selected.log_softmax(-1)
            binary_logit = selected[:, 1] - selected[:, 0]
        else:
            raise ValueError("Label distribution must be full_vocab or yes_no")
        return {
            "log_no": log_probabilities[:, 0],
            "log_yes": log_probabilities[:, 1],
            "p_yes": log_probabilities[:, 1].exp(),
            "binary_logit": binary_logit,
            "answer": selected.argmax(-1),
        }

    def metadata(self):
        return {
            "model": self.config.get("model"),
            "revision": self.config.get("revision"),
            "resolved_revision": getattr(
                self.language_model.config, "_commit_hash", None
            ),
            "labels": self.label_ids,
            "frozen": True,
        }


class SuperG(nn.Module):
    def __init__(self, config, llm=None):
        super().__init__()
        self.config = config
        dimension = int(config.get("dimension", 768))
        cfg = config["encoder"]
        self.history = HistoryEncoder(
            dimension,
            int(cfg.get("heads", 2)),
            int(cfg.get("layers", 1)),
            int(cfg.get("hidden", 3072)),
            int(cfg.get("max_history", 100)),
            float(cfg.get("dropout", 0.1)),
        )
        self.retriever = BehaviorRetriever(
            dimension,
            int(config["retrieval"].get("hidden", 768)),
            float(config["retrieval"].get("weighting", 0.6)),
        )
        self.llm = llm or FrozenBinaryLLM({"dimension": dimension, **config["llm"]})

    def load_llm(self):
        self.llm.load()
        device = next(self.llm.language_model.parameters()).device
        self.history.to(device)
        self.retriever.to(device)
        return device

    def context(
        self, dataset, bank, items, user, item, training=False, ablation="full"
    ):
        cfg = self.config["retrieval"]
        exclude = (
            item
            if training and self.config["data"].get("exclude_query_item", True)
            else None
        )
        positive_histories = [
            dataset.history(index, exclude) for index in range(len(dataset.users))
        ]
        histories = [
            [value for value in dataset.behavior_histories[index] if value != exclude]
            if self.config["data"].get("include_explicit_negative_behaviors", False)
            and positive_histories[index]
            else positive_histories[index]
            for index in range(len(dataset.users))
        ]
        if not positive_histories[user]:
            raise ValueError("Target user has no history after query exclusion")
        values = bank.query(user, item, exclude is not None)
        mode = (
            cfg.get("training_proposal", "semantic")
            if training
            else cfg.get("inference_proposal", "personalized")
        )
        if ablation == "no_ibmm":
            mode = self.config["ablations"].get("no_ibmm_score", "semantic")
        with torch.no_grad():
            candidates = self.retriever.candidates(
                user,
                item,
                values,
                items,
                histories,
                int(cfg.get("similar_users", 5)),
                int(cfg.get("per_user", 5)),
                mode,
                exclude,
                ablation == "no_suem",
            )
        candidates = [
            Behavior(
                behavior.user,
                behavior.item,
                dataset.behavior_labels.get((behavior.user, behavior.item), 1),
            )
            for behavior in candidates
        ]
        sources = list(
            dict.fromkeys([user] + [behavior.user for behavior in candidates])
        )
        fresh = self.history.encode(
            [positive_histories[source] for source in sources], items
        )
        local = {source: index for index, source in enumerate(sources)}
        remapped = [Behavior(local[b.user], b.item, b.label) for b in candidates]
        scores = self.retriever.score(
            local[user],
            item,
            remapped,
            fresh,
            items,
            personalized=ablation != "no_ibmm" or mode == "personalized",
        )
        return {
            "behaviors": candidates,
            "scores": scores,
            "users": {source: fresh[index] for source, index in local.items()},
            "target": user,
            "item": item,
        }

    def prompt(self, dataset, items, context, behaviors, adaptation="learned"):
        cfg = self.config["llm"]
        slots = []
        limit = int(cfg.get("metadata_characters", 256))

        def entity(kind, index):
            if kind == "user":
                identifier = str(dataset.users[index])
                attributes = ""
                vector = context["users"][index]
                marker = "[USER_EMBED]"
            else:
                entry = dataset.items[index]
                identifier = str(entry["id"])
                attributes = " ".join(
                    str(entry.get(key, "")) for key in ("title", "category", "text")
                )[:limit]
                vector = items[index]
                marker = "[ITEM_EMBED]"
            if (
                "[USER_EMBED]" in identifier + attributes
                or "[ITEM_EMBED]" in identifier + attributes
            ):
                raise ValueError("Metadata cannot contain reserved embedding markers")
            if adaptation == "text":
                value = json.dumps(
                    vector.detach().cpu().tolist(), separators=(",", ":")
                )
            else:
                slots.append(vector)
                value = marker
            return f"{kind.title()}: {identifier}; Attributes: {attributes}; Embedding: {value}"

        body = [
            "Target Behavior:",
            entity("user", context["target"]),
            entity("item", context["item"]),
            "Informative Behaviors:",
        ]
        for number, behavior in enumerate(behaviors, 1):
            body.extend(
                [
                    f"Behavior {number}:",
                    entity("user", behavior.user),
                    entity("item", behavior.item),
                    f"User {dataset.users[behavior.user]} interacted with Item {dataset.items[behavior.item]['id']}? {'Yes' if behavior.label else 'No'}",
                ]
            )
        instruction = "Instruction: Based on the above information, predict whether the target user will interact with the new item. Respond with 'Yes' or 'No'."
        if cfg.get("instruction_position", "last") not in {"first", "last"}:
            raise ValueError("Instruction position must be first or last")
        if cfg.get("instruction_position", "last") == "first":
            body.insert(0, instruction)
        else:
            body.append(instruction)
        entities = (
            torch.stack(slots) if slots else items.new_empty((0, items.shape[-1]))
        )
        return "\n".join(body), entities

    def query_loss(self, dataset, bank, items, user, item, label, ablation="full"):
        context = self.context(dataset, bank, items, user, item, True, ablation)
        adaptation = (
            self.config["ablations"].get("no_bhrm_adaptation", "zero_pad")
            if ablation == "no_bhrm"
            else "learned"
        )
        retrieval = items.new_zeros(())
        if ablation != "no_ibmm":
            prompts, slots = zip(
                *(
                    self.prompt(dataset, items, context, [behavior], adaptation)
                    for behavior in context["behaviors"]
                )
            )
            likelihoods = []
            chunk = int(self.config["llm"].get("feedback_batch_size", 1))
            for start in range(0, len(prompts), chunk):
                output = self.llm.score(
                    prompts[start : start + chunk],
                    slots[start : start + chunk],
                    self.config["llm"].get("feedback_distribution", "full_vocab"),
                    adaptation,
                )
                likelihoods.append(output["log_yes"] if label else output["log_no"])
            feedback = torch.cat(likelihoods)
            if self.config["training"].get("detach_feedback", False):
                feedback = feedback.detach()
            retrieval = retrieval_kl(context["scores"], feedback)
        selected = self.retriever.rank(
            context["behaviors"],
            context["scores"],
            int(self.config["retrieval"].get("top_behaviors", 5)),
        )
        prompt, slots = self.prompt(dataset, items, context, selected, adaptation)
        output = self.llm.score([prompt], [slots], adaptation=adaptation)
        recommendation = F.softplus((1 - 2 * label) * output["binary_logit"]).mean()
        requested = (int(self.config["retrieval"].get("similar_users", 5)) + 1) * int(
            self.config["retrieval"].get("per_user", 5)
        )
        return (
            retrieval,
            recommendation,
            {
                "candidates": len(context["behaviors"]),
                "selected": len(selected),
                "source_users": len(
                    {behavior.user for behavior in context["behaviors"]}
                ),
                "requested_candidates": requested,
                "candidate_shortfall": max(0, requested - len(context["behaviors"])),
            },
        )

    def contrastive(self, dataset, items, users, rng):
        users = list(dict.fromkeys(users))
        users = [user for user in users if dataset.histories[user]]
        if len(users) < 2:
            return items.new_zeros(())
        cfg = self.config["contrastive"]
        views, positions = [[], []], [[], []]
        methods = ["mask", "sample", "reorder", "segments"]
        for user in users:
            original = dataset.history(user, max_length=self.history.max_history)
            chosen = rng.choice(methods, 2, replace=False)
            for index, method in enumerate(chosen):
                augmented = augment_history(
                    original,
                    method,
                    rng,
                    float(cfg.get("ratio", 0.2)),
                    int(cfg.get("segments", 3)),
                )
                views[index].append(augmented)
                positions[index].append(
                    [original.index(item) for item in augmented]
                    if method == "mask"
                    else list(range(len(augmented)))
                )
        encoded = [
            self.history.encode(view, items, position)
            for view, position in zip(views, positions)
        ]
        return contrastive_loss(
            *encoded,
            float(cfg.get("temperature", 0.1)),
            cfg.get("denominator", "negative_only"),
        )

    def trainable_state(self):
        return {
            "history": self.history.state_dict(),
            "retriever": self.retriever.state_dict(),
            "adapter": self.llm.adapter.state_dict(),
        }

    def load_trainable_state(self, state):
        self.history.load_state_dict(state["history"])
        self.retriever.load_state_dict(state["retriever"])
        self.llm.adapter.load_state_dict(state["adapter"])
