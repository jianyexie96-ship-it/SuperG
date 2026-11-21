from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Behavior:
    user: int
    item: int
    label: int = 1


class BehaviorRetriever(nn.Module):
    def __init__(self, dimension=768, hidden=768, weighting=0.6):
        super().__init__()
        if not 0 <= weighting <= 1:
            raise ValueError("Retrieval weighting must be in [0,1]")
        self.preference = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, dimension)
        )
        self.weighting = weighting

    def score(self, user, item, behaviors, users, items, personalized=True):
        if not behaviors:
            raise ValueError("No candidate behaviors")
        source = torch.tensor([b.user for b in behaviors], device=users.device)
        historical = torch.tensor([b.item for b in behaviors], device=items.device)
        semantic = F.cosine_similarity(
            users[user].unsqueeze(0), users[source], dim=-1
        ) + F.cosine_similarity(items[item].unsqueeze(0), items[historical], dim=-1)
        if not personalized:
            return semantic
        personal = F.cosine_similarity(
            self.preference(users[user]).unsqueeze(0), items[historical], dim=-1
        )
        return (1 - self.weighting) * semantic + self.weighting * personal

    def candidates(
        self,
        user,
        item,
        users,
        items,
        histories,
        m=5,
        k=5,
        mode="semantic",
        exclude_item=None,
        no_suem=False,
    ):
        if m < 0 or k < 1 or mode not in {"semantic", "personalized"}:
            raise ValueError("Invalid retrieval settings")
        eligible = [
            index
            for index, history in enumerate(histories)
            if index != user and any(value != exclude_item for value in history)
        ]
        if no_suem:
            pool = [
                Behavior(source, historical)
                for source in range(len(histories))
                for historical in histories[source]
                if historical != exclude_item
            ]
            return self.rank(
                pool,
                self.score(user, item, pool, users, items, mode == "personalized"),
                (m + 1) * k,
            )
        if eligible and m:
            similarities = F.cosine_similarity(
                users[user].unsqueeze(0), users[eligible], dim=-1
            )
            order = torch.argsort(similarities, descending=True, stable=True)[
                :m
            ].tolist()
            sources = [user] + [eligible[index] for index in order]
        else:
            sources = [user]
        result = []
        for source in sources:
            pool = [
                Behavior(source, historical)
                for historical in histories[source]
                if historical != exclude_item
            ]
            if pool:
                scores = self.score(
                    user, item, pool, users, items, mode == "personalized"
                )
                result.extend(self.rank(pool, scores, k))
        if not result:
            raise ValueError("No valid history remains after query-item exclusion")
        return result

    @staticmethod
    def rank(behaviors, scores, count):
        order = torch.argsort(scores.detach(), descending=True, stable=True)[
            :count
        ].tolist()
        return [behaviors[index] for index in order]


class UserBank:
    def __init__(self, dataset, encoder, item_features, batch_size=256):
        self.dataset = dataset
        self.encoder = encoder
        self.items = item_features
        self.batch_size = batch_size
        self.values = None

    def _encode(self, users, exclude_item=None):
        result = self.items.new_zeros((len(users), self.items.shape[-1]))
        eligible = [
            (index, self.dataset.history(user, exclude_item))
            for index, user in enumerate(users)
            if self.dataset.history(user, exclude_item)
        ]
        for start in range(0, len(eligible), self.batch_size):
            selected = eligible[start : start + self.batch_size]
            result[[index for index, history in selected]] = self.encoder.encode(
                [history for index, history in selected], self.items
            )
        return result

    def refresh(self):
        training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            self.values = self._encode(list(range(len(self.dataset.users))))
        self.encoder.train(training)

    def query(self, user, item, exclude_query=True):
        if self.values is None:
            self.refresh()
        values = self.values.clone()
        if exclude_query:
            affected = sorted(set(self.dataset.item_users[item]) | {user})
            training = self.encoder.training
            self.encoder.eval()
            with torch.no_grad():
                values[affected] = self._encode(affected, item)
            self.encoder.train(training)
        return values
