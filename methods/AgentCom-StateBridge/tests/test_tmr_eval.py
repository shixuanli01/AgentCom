import pytest

from agentcom.tmr_eval import aggregate_records, shard_indexed_items


def _record(item_index, correct, duration=1.0):
    return {
        "item_index": item_index,
        "prediction": "a" if correct else None,
        "correct": correct,
        "duration": duration,
        "handoffs": [],
    }


def test_aggregate_tmr_records_without_baseline():
    metrics = aggregate_records([_record(0, True), _record(1, False)], {})

    assert metrics["total"] == 2
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["paired_statebridge_items"] == 0
    assert metrics["baseline_accuracy"] is None
    assert metrics["delta_vs_statebridge"] is None
    assert metrics["mcnemar_exact_two_sided_p"] is None


def test_aggregate_tmr_records_with_paired_baseline():
    records = [_record(0, True), _record(1, False)]
    baseline = {
        0: {"prediction": "b", "correct": False, "gold": "a"},
        1: {"prediction": "a", "correct": True, "gold": "a"},
    }

    metrics = aggregate_records(records, baseline)

    assert metrics["paired_statebridge_items"] == 2
    assert metrics["rescues_vs_statebridge"] == 1
    assert metrics["destructions_vs_statebridge"] == 1
    assert metrics["delta_vs_statebridge"] == 0.0
    assert metrics["mcnemar_exact_two_sided_p"] == 1.0


def test_shard_indexed_items_is_balanced_and_lossless():
    items = [(index, {"idx": index}) for index in range(7)]

    shards = [
        shard_indexed_items(items, num_shards=3, shard_index=index)
        for index in range(3)
    ]

    assert [[item_id for item_id, _ in shard] for shard in shards] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]
    assert sorted(item_id for shard in shards for item_id, _ in shard) == list(range(7))


@pytest.mark.parametrize(
    ("num_shards", "shard_index"),
    [(0, 0), (2, -1), (2, 2)],
)
def test_shard_indexed_items_rejects_invalid_topology(num_shards, shard_index):
    with pytest.raises(ValueError):
        shard_indexed_items([], num_shards=num_shards, shard_index=shard_index)
