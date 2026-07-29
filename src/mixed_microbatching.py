"""Deterministic mixed-country optimizer blocks for joint SFT."""

import json
import math
import random
from collections import Counter, defaultdict, deque


ROWS_PER_BLOCK = {
    "indonesian": 9,
    "chinese": 7,
    "sri_lankan": 4,
}
OPTIMIZER_BLOCK_ROWS = sum(ROWS_PER_BLOCK.values())
EXTRA_ZH_SOURCE = "remaining_permutation"


def load_schedule_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                meta = record.get("meta", {})
                rows.append({
                    "messages": record["messages"],
                    "fold": meta.get("fold", -1),
                    "uid": meta.get("uid"),
                    "dataset": meta.get("dataset"),
                    "augmentation_source": meta.get(
                        "augmentation_source", "primary"),
                    "permutation_order": meta.get("permutation_order"),
                })
    return rows


def _balanced_cycle(rows, indices, rng):
    """One coverage-first cycle, with at most one row per UID in each round."""
    by_uid = defaultdict(list)
    for index in indices:
        by_uid[rows[index]["uid"]].append(index)
    for values in by_uid.values():
        rng.shuffle(values)

    order = []
    while by_uid:
        uids = list(by_uid)
        rng.shuffle(uids)
        for uid in uids:
            order.append(by_uid[uid].pop())
            if not by_uid[uid]:
                del by_uid[uid]
    return order


class _UniqueUidPool:
    """Coverage-first row pool that can forbid repeated UIDs in one block."""

    def __init__(self, rows, indices, rng, refill):
        if not indices:
            raise ValueError("cannot build a UID pool from zero rows")
        self.rows = rows
        self.indices = list(indices)
        self.rng = rng
        self.refill = refill
        self.queue = deque(_balanced_cycle(rows, self.indices, rng))

    @property
    def unique_uids(self):
        return len({self.rows[index]["uid"] for index in self.indices})

    def draw(self, count, excluded_uids=None):
        excluded = set(excluded_uids or ())
        if self.unique_uids < count + len(
                excluded.intersection(
                    {self.rows[index]["uid"] for index in self.indices})):
            raise ValueError(
                f"pool has too few UIDs to draw {count} new unique questions")

        selected = []
        deferred = deque()
        while len(selected) < count:
            if not self.queue:
                if deferred:
                    self.queue, deferred = deferred, deque()
                elif self.refill:
                    self.queue.extend(
                        _balanced_cycle(self.rows, self.indices, self.rng))
                else:
                    break
            index = self.queue.popleft()
            uid = self.rows[index]["uid"]
            if uid in excluded:
                deferred.append(index)
                continue
            selected.append(index)
            excluded.add(uid)

        # Deferred rows remain unused and are first candidates next time.
        self.queue.extend(deferred)
        return selected


class _RowPool:
    """Coverage-first pool without a UID uniqueness constraint."""

    def __init__(self, indices, rng):
        if not indices:
            raise ValueError("cannot build a row pool from zero rows")
        self.indices = list(indices)
        self.rng = rng
        self.queue = deque()

    def draw(self, count):
        selected = []
        while len(selected) < count:
            if not self.queue:
                cycle = list(self.indices)
                self.rng.shuffle(cycle)
                self.queue.extend(cycle)
            selected.append(self.queue.popleft())
        return selected


def _split_indices(rows):
    primary = defaultdict(list)
    chinese_extra = []
    for index, row in enumerate(rows):
        dataset = row.get("dataset")
        uid = row.get("uid")
        if dataset not in ROWS_PER_BLOCK:
            raise ValueError(
                f"row {index}: unsupported or missing dataset {dataset!r}")
        if not uid:
            raise ValueError(f"row {index}: missing meta.uid")
        if (dataset == "chinese" and
                row.get("augmentation_source") == EXTRA_ZH_SOURCE):
            chinese_extra.append(index)
        else:
            primary[dataset].append(index)

    missing = [name for name in ROWS_PER_BLOCK if not primary[name]]
    if missing:
        raise ValueError(f"mixed batching is missing primary rows for {missing}")
    if not chinese_extra:
        raise ValueError(
            "mixed batching requires the Chinese remaining-permutation file")
    return primary, chinese_extra


def mixed_optimizer_step_count(rows):
    primary, _ = _split_indices(rows)
    return math.ceil(
        len(primary["indonesian"]) / ROWS_PER_BLOCK["indonesian"])


def build_mixed_optimizer_blocks(rows, seed=42):
    """Build 9-ID/7-ZH/4-SI blocks; Indonesian defines the epoch length."""
    primary, chinese_extra = _split_indices(rows)
    rng = random.Random(seed)
    steps = math.ceil(
        len(primary["indonesian"]) / ROWS_PER_BLOCK["indonesian"])

    id_pool = _UniqueUidPool(
        rows, primary["indonesian"], rng, refill=True)
    zh_primary_pool = _UniqueUidPool(
        rows, primary["chinese"], rng, refill=False)
    zh_extra_pool = _UniqueUidPool(
        rows, chinese_extra, rng, refill=True)
    si_pool = _RowPool(primary["sri_lankan"], rng)

    blocks = []
    for _ in range(steps):
        indonesian = id_pool.draw(ROWS_PER_BLOCK["indonesian"])

        chinese = zh_primary_pool.draw(ROWS_PER_BLOCK["chinese"])
        if len(chinese) < ROWS_PER_BLOCK["chinese"]:
            used_uids = {rows[index]["uid"] for index in chinese}
            chinese.extend(zh_extra_pool.draw(
                ROWS_PER_BLOCK["chinese"] - len(chinese),
                excluded_uids=used_uids,
            ))

        sri_lankan = si_pool.draw(ROWS_PER_BLOCK["sri_lankan"])
        block = indonesian + chinese + sri_lankan
        rng.shuffle(block)
        blocks.append(block)
    return blocks


def mixed_schedule_stats(rows, blocks):
    counts = Counter()
    source_counts = Counter()
    for block in blocks:
        block_counts = Counter(rows[index]["dataset"] for index in block)
        if block_counts != Counter(ROWS_PER_BLOCK):
            raise AssertionError(
                f"invalid mixed optimizer block distribution: {block_counts}")
        for dataset in ("indonesian", "chinese"):
            uids = [rows[index]["uid"] for index in block
                    if rows[index]["dataset"] == dataset]
            if len(uids) != len(set(uids)):
                raise AssertionError(
                    f"{dataset} UID repeated inside optimizer block")
        for index in block:
            dataset = rows[index]["dataset"]
            counts[dataset] += 1
            if dataset == "chinese":
                source_counts[rows[index]["augmentation_source"]] += 1
    return {
        "optimizer_blocks": len(blocks),
        "rows": dict(counts),
        "chinese_sources": dict(source_counts),
    }
