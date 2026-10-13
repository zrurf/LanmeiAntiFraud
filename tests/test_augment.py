import random

import pytest

from train.data.augment import (
    AUGMENT_FACTORIES,
    AUGMENT_REGISTRY,
    build_augment_fn,
    emoji_replace,
    homoglyph_substitute,
    random_space,
)


def test_random_space_inserts_between_every_char():
    assert random_space("诈骗转账", p=1.0) == "诈 骗 转 账 "


def test_random_space_keeps_text_with_zero_probability():
    assert random_space("诈骗转账", p=0.0) == "诈骗转账"


def test_emoji_replace_maps_scam_keywords():
    assert emoji_replace("请转账到安全账户", replace_rate=1.0) == "请💸到🏦"
    assert emoji_replace("请转账", replace_rate=0.0) == "请转账"


def test_build_augment_fn_returns_none_for_empty_kinds():
    assert build_augment_fn([]) is None


def test_build_augment_fn_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_augment_fn(["not_an_operator"])


def test_build_augment_fn_honours_rate_override():
    fn = build_augment_fn(["emoji_replace"], emoji_replace_rate=1.0)
    assert fn("请转账") == "请💸"


def test_sample_rate_zero_keeps_text_untouched():
    fn = build_augment_fn(["random_space"], random_space_rate=1.0, sample_rate=0.0)
    assert [fn("诈骗转账") for _ in range(50)] == ["诈骗转账"] * 50


def test_sample_rate_one_applies_to_every_sample():
    fn = build_augment_fn(["random_space"], random_space_rate=1.0, sample_rate=1.0)
    assert [fn("诈骗转账") for _ in range(50)] == ["诈 骗 转 账 "] * 50


def test_sample_rate_is_bernoulli():
    random.seed(0)
    fn = build_augment_fn(["random_space"], random_space_rate=1.0, sample_rate=0.3)
    changed = sum(1 for _ in range(3000) if fn("诈骗转账") == "诈 骗 转 账 ")
    assert 0.26 < changed / 3000 < 0.34


def test_sample_rate_out_of_range_raises():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            build_augment_fn(["random_space"], sample_rate=bad)


def test_sample_rate_reaches_all_operators_in_chain():
    """抽样是对整条链生效，不是逐个算子抽"""
    random.seed(1)
    fn = build_augment_fn(
        ["emoji_replace", "random_space"],
        emoji_replace_rate=1.0,
        random_space_rate=1.0,
        sample_rate=0.5,
    )
    outputs = {fn("请转账") for _ in range(200)}
    assert outputs == {"请转账", "请 💸 "}


def test_homoglyph_table_is_well_formed():
    """字表逐对人工核定，键值都必须是单个不同的汉字，替代字不得重复

    这条测试守着之前踩过的坑：旧表 10 对里有 8 对无法通过相似度闸门而静默失效
    """
    from train.data.augment import _HOMOGLYPH_CANDIDATES

    assert len(_HOMOGLYPH_CANDIDATES) >= 50, "字表规模过小，覆盖不了诈骗词表"
    for src, dst in _HOMOGLYPH_CANDIDATES.items():
        assert len(src) == 1 and len(dst) == 1, f"必须是单个汉字：{src!r} → {dst!r}"
        assert not src.isascii() and not dst.isascii(), f"必须是汉字：{src!r} → {dst!r}"
        assert src != dst, f"替代字不能与原字相同：{src!r}"
    values = list(_HOMOGLYPH_CANDIDATES.values())
    assert len(set(values)) == len(values), "同一个替代字被多个原字占用"


def test_homoglyph_substitute_fires_without_gate():
    """默认不做相似度筛选，字表覆盖到的字必须真的被替换"""
    from train.data.augment import _HOMOGLYPH_CANDIDATES

    covered = "".join(_HOMOGLYPH_CANDIDATES)
    assert homoglyph_substitute(covered, p=1.0) != covered
    assert homoglyph_substitute(covered, p=0.0) == covered
    assert homoglyph_substitute("户", p=1.0) == "戶"


def test_homoglyph_optional_gate_rejects_real_variants():
    """开启旧闸门时，繁简/异体这类真同形字会被误挡——用测试记录该指标的这一局限"""
    assert homoglyph_substitute("户", p=1.0, min_similarity=0.7) == "户"


def test_registry_exposes_registered_operators():
    assert "random_space" in AUGMENT_FACTORIES
    assert set(AUGMENT_REGISTRY.keys()) == set(AUGMENT_FACTORIES)
    assert callable(AUGMENT_REGISTRY["random_space"])


def test_registry_unknown_key_returns_default():
    assert AUGMENT_REGISTRY.get("not_an_operator") is None