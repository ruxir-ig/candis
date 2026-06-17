"""Verify the ranking metrics against hand-computed values."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import ndcg_at_k, precision_at_k, average_precision  # noqa: E402


def almost(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_ndcg_perfect_order_is_one():
    rels = [4, 3, 2, 1, 0]
    assert almost(ndcg_at_k(rels, 5), 1.0)


def test_ndcg_known_value():
    # rels = [3, 2, 3, 0, 1, 2]; DCG@6 / IDCG@6 -> 0.948811
    rels = [3, 2, 3, 0, 1, 2]
    assert almost(ndcg_at_k(rels, 6), 0.948811, tol=1e-4)


def test_ndcg_worst_order_below_one():
    assert ndcg_at_k([0, 1, 2, 3, 4], 5) < 1.0


def test_precision_at_k():
    rels = [4, 0, 3, 2, 3]  # relevant = grade>=3 -> positions 1,3,5
    assert almost(precision_at_k(rels, 5), 3 / 5)
    assert almost(precision_at_k(rels, 1), 1.0)


def test_average_precision():
    # relevant at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
    rels = [3, 0, 4, 0]
    assert almost(average_precision(rels), (1.0 + 2 / 3) / 2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("All metric tests passed.")
