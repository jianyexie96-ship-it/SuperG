import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_cold_split_and_query_exclusion(tmp_path):
    from data import RecommendationData, prepare_dataset

    items = [{"id": str(i), "text": f"item {i}"} for i in range(20)]
    rows = [
        {"user": str(u), "item": str(i), "timestamp": i}
        for u in range(3)
        for i in range(20)
    ]
    path = prepare_dataset(rows, items, tmp_path / "data.json", seed=7)
    dataset = RecommendationData(path, require_features=False)
    pools = [set(dataset.item_pools[s]) for s in ("train", "val", "test")]
    assert (
        not pools[0] & pools[1] and not pools[0] & pools[2] and not pools[1] & pools[2]
    )
    target = dataset.item_pools["train"][0]
    assert all(target not in dataset.history(u, exclude_item=target) for u in range(3))
    assert len(dataset.eligible_users("test")) == 3
    negatives = dataset.sample_training_pairs(1, 7)
    assert all(item in pools[0] for _, item, _ in negatives)


def test_feature_mean_uses_only_present_modalities():
    from features import combine_modalities

    values = {
        "visual": np.array([[2.0, 0.0], [0.0, 0.0]]),
        "text": np.array([[0.0, 2.0], [4.0, 6.0]]),
    }
    masks = {"visual": np.array([True, False]), "text": np.array([True, True])}
    assert np.allclose(combine_modalities(values, masks), [[1, 1], [4, 6]])
    with pytest.raises(ValueError):
        combine_modalities(values, {k: np.zeros(2, bool) for k in values})


def test_literal_contrastive_formula_and_batch_identity():
    from model import contrastive_loss

    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(
        contrastive_loss(a, a, temperature=1, denominator="negative_only"),
        torch.tensor(-2.0),
    )
    assert contrastive_loss(a, a, temperature=1, denominator="include_positive") > 0
    with pytest.raises(ValueError):
        contrastive_loss(a[:1], a[:1])


def test_retriever_formula_topk_and_global_ranking():
    from retrieval import BehaviorRetriever

    users = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    items = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    retriever = BehaviorRetriever(2, 4, 0.6)
    candidates = retriever.candidates(
        0,
        0,
        users,
        items,
        [[1, 2], [0, 1], [0, 2]],
        m=1,
        k=1,
        mode="semantic",
        exclude_item=0,
    )
    assert {b.user for b in candidates} == {0, 2}
    assert all(b.item != 0 for b in candidates)
    assert len(candidates) == 2
    scores = retriever.score(0, 0, candidates, users, items, personalized=False)
    expected = torch.stack(
        [
            torch.cosine_similarity(users[0], users[b.user], dim=0)
            + torch.cosine_similarity(items[0], items[b.item], dim=0)
            for b in candidates
        ]
    )
    assert torch.allclose(scores, expected)
    ranked = retriever.rank(candidates, scores, 1)
    assert ranked[0] == candidates[int(scores.argmax())]


def test_kl_direction_and_both_gradient_paths():
    from model import retrieval_kl

    scores = torch.tensor([1.0, 0.0, -0.5], requires_grad=True)
    feedback = torch.tensor([-0.3, -0.8, -0.6], requires_grad=True)
    loss = retrieval_kl(scores, feedback)
    p = scores.softmax(0)
    expected = (p * (scores.log_softmax(0) - feedback.log_softmax(0))).sum()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert scores.grad.abs().sum() > 0 and feedback.grad.abs().sum() > 0


def test_four_history_augmentations_preserve_real_items():
    from model import augment_history

    for name in ("mask", "sample", "reorder", "segments"):
        result = augment_history(
            list(range(10)), name, np.random.default_rng(3), 0.5, 3
        )
        assert result and set(result).issubset(set(range(10)))
        assert len(result) <= 10


def tiny_lm():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = ["<pad>", "<s>", "</s>", "<unk>", "Yes", "No", "user", "assistant"]
    backend = Tokenizer(
        WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = "{{ bos_token }}{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    config = LlamaConfig(
        vocab_size=len(words),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return LlamaForCausalLM(config), tokenizer


def test_real_frozen_lm_repeated_entity_injection_is_differentiable():
    from model import FrozenBinaryLLM

    lm, tokenizer = tiny_lm()
    config = {
        "dimension": 4,
        "adapter_hidden": 8,
        "max_length": 512,
        "prediction_distribution": "yes_no",
        "feedback_distribution": "full_vocab",
        "instruction_position": "last",
    }
    scorer = FrozenBinaryLLM(config, model=lm, tokenizer=tokenizer)
    embeddings = torch.randn(4, 4, requires_grad=True)
    with torch.no_grad():
        scorer.adapter[0].bias.fill_(10)
    assert (scorer.adapter[0](embeddings) > 0).all()
    prompt = "Target [USER_EMBED] [ITEM_EMBED] History [USER_EMBED] [ITEM_EMBED] Yes"
    output = scorer.score([prompt], [embeddings], distribution="full_vocab")
    assert output["p_yes"].shape == (1,)
    (-output["log_yes"].mean()).backward()
    assert embeddings.grad is not None and torch.all(embeddings.grad.abs().sum(1) > 0)
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in lm.parameters()
    )
    assert scorer.adapter[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="slot|marker"):
        scorer.score([prompt], [embeddings[:2]])


def test_history_encoder_masked_mean_and_empty_rejection():
    from model import HistoryEncoder

    encoder = HistoryEncoder(4, 2, 1, 8, 8, 0)
    z = torch.randn(5, 4)
    assert encoder.encode([[1, 2], [3]], z).shape == (2, 4)
    with pytest.raises(ValueError):
        encoder.encode([[]], z)


def test_full_cold_ranking_metrics_and_coverage():
    from evaluate import ranking_metrics

    result = ranking_metrics({0: [2, 3], 1: [3, 2]}, {0: {2}, 1: {2}}, [1, 2])
    assert result["Recall@1"] == 0.5 and result["Recall@2"] == 1
    assert result["NDCG@2"] == pytest.approx((1 + 1 / np.log2(3)) / 2)
    with pytest.raises(ValueError):
        ranking_metrics({0: [2]}, {0: {2}, 1: {2}}, [1])


def small_config():
    return {
        "dimension": 4,
        "data": {"exclude_query_item": True},
        "encoder": {
            "heads": 2,
            "layers": 1,
            "hidden": 8,
            "max_history": 8,
            "dropout": 0,
        },
        "retrieval": {
            "hidden": 8,
            "weighting": 0.6,
            "similar_users": 1,
            "per_user": 1,
            "top_behaviors": 1,
            "training_proposal": "semantic",
            "inference_proposal": "personalized",
        },
        "llm": {
            "adapter_hidden": 8,
            "max_length": 512,
            "feedback_distribution": "full_vocab",
            "prediction_distribution": "yes_no",
            "feedback_batch_size": 1,
        },
        "training": {"detach_feedback": False, "seed": 1},
        "contrastive": {
            "ratio": 0.3,
            "segments": 2,
            "temperature": 0.1,
            "denominator": "negative_only",
        },
        "ablations": {"no_ibmm_score": "semantic", "no_bhrm_adaptation": "zero_pad"},
        "evaluation": {"ks": [1, 2]},
    }


@pytest.fixture
def small_dataset(tmp_path):
    from data import RecommendationData, prepare_dataset
    from features import save_features

    histories = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    rows = [
        {"user": str(u), "item": str(item), "split": "train"}
        for u, history in enumerate(histories)
        for item in history
    ]
    rows += [
        {"user": str(u), "item": str(item), "split": split}
        for u in range(3)
        for item, split in ((6, "val"), (7, "test"))
    ]
    path = prepare_dataset(
        rows,
        [{"id": str(i), "text": f"description {i}"} for i in range(8)],
        tmp_path / "data.json",
    )
    save_features(
        tmp_path / "features.npz",
        [str(i) for i in range(8)],
        {"text": np.random.default_rng(3).normal(size=(8, 4)).astype(np.float32)},
        dimension=4,
    )
    return RecommendationData(path)


def test_real_superg_feedback_loss_reaches_shared_adapter_without_lm_gradients(
    small_dataset,
):
    from model import FrozenBinaryLLM, SuperG
    from retrieval import UserBank

    cfg = small_config()
    lm, tokenizer = tiny_lm()
    scorer = FrozenBinaryLLM({"dimension": 4, **cfg["llm"]}, lm, tokenizer)
    model = SuperG(cfg, scorer)
    items = torch.tensor(small_dataset.features)
    bank = UserBank(small_dataset, model.history, items, 2)
    context = model.context(small_dataset, bank, items, 0, 0, training=True)
    assert all(behavior.item != 0 for behavior in context["behaviors"])
    retrieval, recommendation, coverage = model.query_loss(
        small_dataset, bank, items, 0, 0, 1
    )
    assert torch.isfinite(retrieval) and torch.isfinite(recommendation)
    retrieval.backward()
    assert scorer.adapter[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in lm.parameters())
    assert coverage["candidates"] == 2


def test_all_ablations_have_real_probability_paths(small_dataset):
    from model import FrozenBinaryLLM, SuperG
    from retrieval import UserBank

    cfg = small_config()
    lm, tokenizer = tiny_lm()
    model = SuperG(cfg, FrozenBinaryLLM({"dimension": 4, **cfg["llm"]}, lm, tokenizer))
    items = torch.tensor(small_dataset.features)
    bank = UserBank(small_dataset, model.history, items, 2)
    for name in ("no_suem", "no_ibmm", "no_bhrm"):
        retrieval, recommendation, _coverage = model.query_loss(
            small_dataset, bank, items, 0, 0, 1, name
        )
        assert torch.isfinite(recommendation)
        if name == "no_ibmm":
            assert retrieval == 0


def test_checkpoint_records_the_trained_ablation(small_dataset, tmp_path):
    from model import FrozenBinaryLLM, SuperG
    from train import save_checkpoint

    cfg = small_config()
    lm, tokenizer = tiny_lm()
    model = SuperG(cfg, FrozenBinaryLLM({"dimension": 4, **cfg["llm"]}, lm, tokenizer))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, optimizer, 0, 0, 0.0, 0, small_dataset, cfg, [], ablation="no_suem"
    )
    state = torch.load(path, weights_only=False)
    assert state["ablation"] == "no_suem"


def test_precomputed_feature_import_respects_explicit_availability(
    small_dataset, tmp_path
):
    from features import import_features

    visual = tmp_path / "visual.npy"
    text = tmp_path / "text.npy"
    mask = tmp_path / "mask.npy"
    np.save(visual, np.ones((8, 4), np.float32) * 100)
    np.save(text, np.ones((8, 4), np.float32) * 2)
    availability = np.ones(8, bool)
    availability[0] = False
    np.save(mask, availability)
    output = tmp_path / "imported.npz"
    import_features(
        small_dataset,
        {"visual": str(visual), "text": str(text)},
        output,
        4,
        {"visual": str(mask)},
    )
    with np.load(output) as archive:
        assert np.allclose(archive["z"][0], 2)
        assert np.allclose(archive["z"][1], 51)
        assert not archive["available_visual"][0]


def test_feature_cli_accepts_availability_masks():
    from main import parser

    args = parser().parse_args(
        [
            "features",
            "--data",
            "data.json",
            "--output",
            "features.npz",
            "--visual-mask-npy",
            "mask.npy",
        ]
    )
    assert args.visual_mask_npy == "mask.npy"


def test_prepared_data_rejects_heldout_edge_in_training_history(small_dataset):
    from data import RecommendationData

    payload = json.loads(small_dataset.path.read_text())
    heldout = next(row for row in payload["interactions"] if row["split"] == "test")
    heldout["split"] = "train"
    small_dataset.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pool|cold|split"):
        RecommendationData(small_dataset.path)


def test_five_seed_sweep_only_writes_jobs_without_execution(small_dataset, tmp_path):
    from types import SimpleNamespace

    from main import sweep_jobs

    cfg = small_config()
    cfg["sweeps"] = {"retrieval.weighting": [0.6, 0.8]}
    args = SimpleNamespace(
        output=str(tmp_path / "sweep"),
        data=str(small_dataset.path),
        features=None,
        seeds=[1, 2, 3, 4, 5],
        execute=False,
    )
    jobs = sweep_jobs(args, cfg)
    assert len(jobs) == 25
    assert {job["seed"] for job in jobs} == {1, 2, 3, 4, 5}
    assert all(
        "--ablation" in job["train"] and "--ablation" in job["evaluate"] for job in jobs
    )


def test_blip_vision_loader_recovers_caption_checkpoint_weights(tmp_path):
    from transformers import (
        BlipConfig,
        BlipForConditionalGeneration,
        BlipTextConfig,
        BlipVisionConfig,
    )

    from features import load_vision_encoder

    text = BlipTextConfig(
        vocab_size=12,
        hidden_size=8,
        encoder_hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
    )
    vision = BlipVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    model = BlipForConditionalGeneration(
        BlipConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    )
    with torch.no_grad():
        model.vision_model.post_layernorm.weight.fill_(0.77)
    folder = tmp_path / "blip"
    model.save_pretrained(folder)
    loaded = load_vision_encoder(str(folder), local_files_only=True)
    assert torch.equal(
        loaded.post_layernorm.weight, model.vision_model.post_layernorm.weight
    )


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_audio_loading_preserves_signal_and_averages_stereo(tmp_path, sample_rate):
    import wave

    from features import read_audio

    samples = np.tile(np.array([[8192, 0]], dtype="<i2"), (320, 1))
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    signal = read_audio(path, max_seconds=0.01)
    assert signal.shape == (1, 160)
    assert torch.allclose(
        signal[:, 30:-30], torch.full_like(signal[:, 30:-30], 0.125), atol=2e-4
    )
    with pytest.raises(ValueError, match="duration|positive"):
        read_audio(path, max_seconds=-1)


@pytest.mark.parametrize("distribution", ["yes_no", "full_vocab"])
@pytest.mark.parametrize("gap", [-80.0, -20.0, 20.0, 80.0])
def test_binary_score_preserves_extreme_log_odds_and_token_feedback(
    monkeypatch, distribution, gap
):
    from model import FrozenBinaryLLM

    language_model, tokenizer = tiny_lm()
    scorer = FrozenBinaryLLM(
        {"dimension": 4, "max_length": 512}, language_model, tokenizer
    )
    no_id, yes_id = scorer.label_ids
    logits = torch.full((1, len(tokenizer)), -1000.0)
    logits[0, no_id] = 0
    if distribution == "full_vocab":
        logits[0, tokenizer.unk_token_id] = 2
    others = logits.clone()
    others[:, yes_id] = -torch.inf
    offset = others.logsumexp(-1) if distribution == "full_vocab" else logits[:, no_id]
    logits[:, yes_id] = gap + offset
    logits.requires_grad_()

    def forward(inputs_embeds, **kwargs):
        return SimpleNamespace(
            logits=logits[:, None].expand(-1, inputs_embeds.shape[1], -1)
        )

    monkeypatch.setattr(language_model, "forward", forward)
    output = scorer.score(["user"], [torch.empty(0, 4)], distribution=distribution)
    torch.testing.assert_close(
        output["binary_logit"], torch.tensor([gap]), atol=0, rtol=1e-6
    )
    feedback = (
        logits.log_softmax(-1)[:, [no_id, yes_id]]
        if distribution == "full_vocab"
        else logits[:, [no_id, yes_id]].log_softmax(-1)
    )
    torch.testing.assert_close(output["log_no"], feedback[:, 0])
    torch.testing.assert_close(output["log_yes"], feedback[:, 1])
    torch.testing.assert_close(output["p_yes"], feedback[:, 1].exp())
    if distribution == "full_vocab":
        log_not_yes = -torch.nn.functional.softplus(output["binary_logit"])
        assert not torch.allclose(output["log_no"], log_not_yes)
    target = float(gap < 0)
    loss = torch.nn.functional.softplus(
        (1 - 2 * target) * output["binary_logit"]
    ).mean()
    gradient = torch.autograd.grad(loss, logits)[0][0, yes_id]
    torch.testing.assert_close(
        gradient, torch.tensor(1 - 2 * target), atol=0, rtol=1e-6
    )


@pytest.mark.parametrize("gap", [-80.0, -20.0, 20.0, 80.0])
@pytest.mark.parametrize("label", [0, 1])
def test_query_bce_preserves_extreme_loss_and_adapter_gradient(monkeypatch, gap, label):
    from model import SuperG

    class TensorScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor([gap]))

        def score(self, *args, **kwargs):
            return {"p_yes": self.adapter.sigmoid(), "binary_logit": self.adapter}

    scorer = TensorScorer()
    model = SuperG(small_config(), scorer)
    monkeypatch.setattr(
        model, "context", lambda *args: {"behaviors": [], "scores": torch.empty(0)}
    )
    monkeypatch.setattr(model, "prompt", lambda *args: ("fixture", torch.empty(0, 4)))
    _, loss, _ = model.query_loss(None, None, torch.zeros(1, 4), 0, 0, label, "no_ibmm")
    signed = (1 - 2 * label) * gap
    expected_loss = torch.tensor(np.logaddexp(0, signed), dtype=torch.float32)
    expected_gradient = torch.tensor(
        (1 - 2 * label) / (1 + np.exp(-signed)), dtype=torch.float32
    )
    torch.testing.assert_close(loss, expected_loss, atol=0, rtol=1e-6)
    gradient = torch.autograd.grad(loss, scorer.adapter)[0][0]
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=1e-6)
