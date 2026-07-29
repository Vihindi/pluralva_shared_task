"""Dependency-free hybrid, language-homogeneous training schedules."""

import json
import random
from collections import defaultdict


EXPECTED_UID_ROWS = {
    "chinese": 4,
    "indonesian": 20,
    "sri_lankan": 2,
}


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
                })
    return rows


def _uid_prompt_buckets(rows, indices, rng):
    """Split one UID into prompt-identical augmentation/vote packets."""
    prompt_buckets = defaultdict(list)
    for index in indices:
        prompt = tuple(
            (message.get("role"), message.get("content"))
            for message in rows[index]["messages"][:-1]
        )
        prompt_buckets[prompt].append(index)

    buckets = list(prompt_buckets.values())
    rng.shuffle(buckets)
    for bucket in buckets:
        rng.shuffle(bucket)
    return buckets


def _build_indonesian_blocks(uid_buckets):
    """Four UIDs/step, one complete five-vote permutation from each UID."""
    count = len(uid_buckets)
    if count < 4:
        raise ValueError("hybrid Indonesian batching requires at least 4 UIDs")
    blocks = []
    for step in range(count):
        block = []
        for permutation_slot in range(4):
            uid_position = (step + permutation_slot) % count
            bucket = uid_buckets[uid_position][permutation_slot]
            block.extend(bucket)
        blocks.append(block)
    return blocks


def _build_chinese_blocks(uid_buckets):
    """Twenty distinct UIDs/step, balanced across four permutation slots."""
    count = len(uid_buckets)
    if count % 5:
        raise ValueError(
            f"hybrid Chinese batching needs a UID count divisible by 5; got "
            f"{count}")
    blocks = []
    for step in range(count // 5):
        block = []
        start = step * 5
        for permutation_slot in range(4):
            offset = permutation_slot * 5
            for item in range(5):
                uid_position = (start + offset + item) % count
                block.extend(uid_buckets[uid_position][permutation_slot])
        blocks.append(block)
    return blocks


def _build_sri_lankan_blocks(uid_buckets, optimizer_block_rows):
    """Keep each A/B pair together and pack ten UIDs into a full step."""
    blocks = []
    current = []
    for buckets in uid_buckets:
        packet = [index for bucket in buckets for index in bucket]
        if current and len(current) + len(packet) > optimizer_block_rows:
            blocks.append(current)
            current = []
        current.extend(packet)
        if len(current) == optimizer_block_rows:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def build_grouped_optimizer_blocks(rows, optimizer_block_rows=20, seed=42):
    """Build the 20-row hybrid schedule, with one language per update."""
    if optimizer_block_rows != 20:
        raise ValueError(
            "the hybrid schedule currently requires optimizer_block_rows=20")

    rng = random.Random(seed)
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        dataset = row.get("dataset")
        uid = row.get("uid")
        if dataset not in EXPECTED_UID_ROWS:
            raise ValueError(
                f"row {index}: unsupported or missing dataset {dataset!r}")
        if not uid:
            raise ValueError(f"row {index}: missing meta.uid")
        grouped[dataset][uid].append(index)

    blocks_by_dataset = {}
    partial_blocks = []
    for dataset in sorted(grouped):
        expected_size = EXPECTED_UID_ROWS[dataset]
        uid_buckets = []
        for uid, indices in grouped[dataset].items():
            if len(indices) != expected_size:
                raise ValueError(
                    f"{dataset} UID {uid!r} has {len(indices)} rows; expected "
                    f"{expected_size}. Rebuild full SFT data with --n_perms 4, "
                    "--id_mode probability, and --si_mode binary.")
            buckets = _uid_prompt_buckets(rows, indices, rng)
            expected_bucket_sizes = {
                "chinese": [1, 1, 1, 1],
                "indonesian": [5, 5, 5, 5],
                "sri_lankan": [1, 1],
            }[dataset]
            if sorted(map(len, buckets)) != expected_bucket_sizes:
                raise ValueError(
                    f"{dataset} UID {uid!r} has prompt-bucket sizes "
                    f"{sorted(map(len, buckets))}; expected "
                    f"{expected_bucket_sizes}")
            uid_buckets.append(buckets)

        rng.shuffle(uid_buckets)
        if dataset == "indonesian":
            dataset_blocks = _build_indonesian_blocks(uid_buckets)
        elif dataset == "chinese":
            dataset_blocks = _build_chinese_blocks(uid_buckets)
        else:
            dataset_blocks = _build_sri_lankan_blocks(
                uid_buckets, optimizer_block_rows)

        full_blocks = [block for block in dataset_blocks
                       if len(block) == optimizer_block_rows]
        partial_blocks.extend(
            (dataset, block) for block in dataset_blocks
            if len(block) != optimizer_block_rows)
        blocks_by_dataset[dataset] = full_blocks

    counts = {dataset: len(blocks)
              for dataset, blocks in blocks_by_dataset.items()}
    used = {dataset: 0 for dataset in counts}
    total = sum(counts.values())
    tie_order = list(counts)
    rng.shuffle(tie_order)
    tie_rank = {dataset: rank for rank, dataset in enumerate(tie_order)}
    interleaved = []
    for position in range(total):
        available = [dataset for dataset in counts
                     if used[dataset] < counts[dataset]]
        dataset = max(
            available,
            key=lambda name: (
                (position + 1) * counts[name] / total - used[name],
                -tie_rank[name],
            ),
        )
        interleaved.append(blocks_by_dataset[dataset][used[dataset]])
        used[dataset] += 1

    partial_blocks.sort(key=lambda item: (item[0], len(item[1])))
    interleaved.extend(block for _, block in partial_blocks)
    return interleaved


def grouped_schedule_stats(rows, blocks):
    stats = defaultdict(lambda: {"rows": 0, "blocks": 0, "uids": set()})
    for block in blocks:
        languages = {rows[index]["dataset"] for index in block}
        if len(languages) != 1:
            raise AssertionError(f"mixed-language optimizer block: {languages}")
        dataset = next(iter(languages))
        stats[dataset]["rows"] += len(block)
        stats[dataset]["blocks"] += 1
        stats[dataset]["uids"].update(rows[index]["uid"] for index in block)
    return {
        dataset: {
            "rows": values["rows"],
            "optimizer_blocks": values["blocks"],
            "uids": len(values["uids"]),
        }
        for dataset, values in stats.items()
    }
