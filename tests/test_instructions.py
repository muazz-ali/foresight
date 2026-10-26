"""Language templates must name apple vs tangerine distinctly."""

from __future__ import annotations

from policy.features import default_language


def test_apple_vs_tangerine_differ():
    a = default_language("apple12", "bowl", seed=0)
    t = default_language("tangerine03", "bowl", seed=0)
    assert "apple" in a.lower()
    assert "tangerine" in t.lower()
    assert a != t


def test_default_fallback_mentions_object():
    s = default_language("can", "basket", seed=1)
    assert "can" in s.lower()
    assert "basket" in s.lower() or "bowl" in s.lower()


def test_same_object_has_more_than_one_wording():
    from policy.features import paraphrase_task

    seen = {default_language("apple", "bowl", seed=i) for i in range(20)}
    assert len(seen) >= 3
    # A saved sentence can be said another way, but it stays an apple → bowl.
    other = paraphrase_task("Pick the red apple and place it in the bowl.", seed=3)
    assert "apple" in other.lower()
    assert "tangerine" not in other.lower()
    assert "bowl" in other.lower()


def test_paraphrase_does_not_turn_tangerine_into_orange():
    from policy.features import paraphrase_task

    text = paraphrase_task(
        "Pick the orange tangerine and put it in the box.", seed=2
    )
    assert "tangerine" in text.lower()
    assert "box" in text.lower()
