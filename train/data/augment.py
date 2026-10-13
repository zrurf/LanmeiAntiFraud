from __future__ import annotations

import logging
import random
import threading
from functools import lru_cache, partial
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "emoji_replace",
    "emoji_mood_enhance",
    "emoji_strip",
    "homoglyph_substitute",
    "random_space",
    "homoglyph_edit_distance",
    "homoglyph_distance_check",
    "build_homophone",
    "build_random_delete",
    "build_char_swap",
    "build_random_entity",
    "build_augment_fn",
    "AUGMENT_FACTORIES",
    "AUGMENT_REGISTRY",
]

_emoji_lib = None
_mood_to_emoji_fn = None
_char_similarity_cls = None
_nlpcda_mod = None
_homoglyphs_mod = None
_init_lock = threading.Lock()


def _get_emoji_lib():
    global _emoji_lib
    if _emoji_lib is None:
        with _init_lock:
            if _emoji_lib is None:
                import emoji as _em  # type: ignore
                _emoji_lib = _em
    return _emoji_lib


def _get_mood_to_emoji() -> Callable[[str], str]:
    global _mood_to_emoji_fn
    if _mood_to_emoji_fn is None:
        with _init_lock:
            if _mood_to_emoji_fn is None:
                from moodemoji import mood_to_emoji as _m  # type: ignore
                _mood_to_emoji_fn = _m
    return _mood_to_emoji_fn


def _get_char_similarity_cls():
    global _char_similarity_cls
    if _char_similarity_cls is None:
        with _init_lock:
            if _char_similarity_cls is None:
                from char_similar_z import CharSimilarity as _CS  # type: ignore
                _char_similarity_cls = _CS
    return _char_similarity_cls


@lru_cache(maxsize=1)
def _get_char_sim():
    return _get_char_similarity_cls()()


def _get_nlpcda():
    global _nlpcda_mod
    if _nlpcda_mod is None:
        with _init_lock:
            if _nlpcda_mod is None:
                from nlpcda import (  # type: ignore
                    CharPositionExchange,
                    Homophone,
                    RandomDeleteChar,
                    Randomword,
                )
                _nlpcda_mod = {
                    "Homophone": Homophone,
                    "RandomDeleteChar": RandomDeleteChar,
                    "CharPositionExchange": CharPositionExchange,
                    "Randomword": Randomword,
                }
    return _nlpcda_mod


def _get_homoglyphs():
    global _homoglyphs_mod
    if _homoglyphs_mod is None:
        with _init_lock:
            if _homoglyphs_mod is None:
                from HomoglyphsCJK import (  # type: ignore
                    download_dict,
                    hg_distance,
                )
                _homoglyphs_mod = {
                    "download_dict": download_dict,
                    "hg_distance": hg_distance,
                }
    return _homoglyphs_mod


@lru_cache(maxsize=4)
def _ensure_homoglyph_dict_loaded(lang: str) -> bool:
    _get_homoglyphs()["download_dict"](lang)
    return True


_SCAM_EMOJI_MAP: dict[str, str] = {
    "钱": "💰", "转账": "💸", "安全账户": "🏦", "冻结": "🧊",
    "紧急": "🚨", "恭喜": "🎉", "验证码": "🔢", "身份证": "🪪",
    "银行卡": "💳", "中奖": "🎁", "红包": "🧧", "投资": "📈",
    "返利": "💹", "刷单": "🛒", "客服": "🎧", "退款": "↩️",
    "扫码": "📱", "链接": "🔗", "密码": "🔑", "账户": "🏦",
}

_SCAM_EMOJI_KEYS_SORTED: tuple[str, ...] = tuple(
    sorted(_SCAM_EMOJI_MAP, key=len, reverse=True)
)

_CN_MOOD_MAP: dict[str, str] = {
    "开心": "happy", "高兴": "happy", "紧急": "anxious",
    "害怕": "fearful", "恭喜": "celebratory", "愤怒": "angry",
    "伤心": "sad", "惊喜": "surprised", "安心": "relieved",
}

_PROTECTED_WORDS: frozenset[str] = frozenset(_SCAM_EMOJI_MAP) | frozenset(_CN_MOOD_MAP)

_PROTECTED_AUGMENT_NAMES: frozenset[str] = frozenset(
    {"random_delete", "char_swap", "random_entity", "homophone"}
)

# 同形字替换表
_HOMOGLYPH_CANDIDATES: dict[str, str] = {
    # 钱款资金类（贝/貝、钅/金 部件差异）
    "账": "賬", "银": "銀", "钱": "錢", "资": "資", "财": "財", "贷": "貸",
    "费": "費", "质": "質", "购": "購", "货": "貨", "赚": "賺", "贴": "貼",
    "赏": "賞", "赔": "賠", "贸": "貿", "贯": "貫", "贫": "貧",
    # 言语通知类（讠/訁 部件差异）
    "证": "証", "诉": "訴", "话": "話", "说": "説", "请": "請", "认": "認",
    "谈": "談", "调": "調", "询": "詢", "讯": "訊", "谢": "謝", "诚": "誠",
    "误": "誤", "记": "記", "计": "計",
    # 马部/车部/门部/见部（马/馬、车/車、门/門、见/見）
    "验": "験", "骗": "騙", "驻": "駐",
    "转": "転", "库": "庫", "连": "連", "军": "軍",
    "门": "門", "问": "問", "间": "間", "闻": "聞", "阅": "閱", "闲": "閑",
    "见": "見", "规": "規", "观": "觀",
    # 页部（页/頁）
    "额": "額", "领": "領", "项": "項", "题": "題", "预": "預", "顺": "順",
    # 单字与常见形近字
    "单": "単", "户": "戶", "东": "東", "车": "車", "马": "馬", "买": "買",
    "微": "徽", "信": "伩", "返": "反", "款": "欸", "刷": "唰", "学": "茡",
    "教": "敎", "务": "務", "检": "檢", "密": "蜜", "客": "容", "安": "宇",
    "保": "促", "全": "金", "取": "耶", "友": "右", "局": "届",
}

@lru_cache(maxsize=512)
def _cached_shape_similarity(ch: str, candidate: str) -> float:
    return _get_char_sim().std_cal_sim(ch, candidate, kind="shape")


def emoji_replace(text: str, replace_rate: float = 0.6) -> str:
    if not text:
        return text
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        for word in _SCAM_EMOJI_KEYS_SORTED:
            if text.startswith(word, i):
                if random.random() < replace_rate:
                    parts.append(_SCAM_EMOJI_MAP[word])
                else:
                    parts.append(word)
                i += len(word)
                break
        else:
            parts.append(text[i])
            i += 1
    return "".join(parts)


def emoji_mood_enhance(text: str, p: float = 0.06) -> str:
    mood_to_emoji = _get_mood_to_emoji()
    for cn_word, en_mood in _CN_MOOD_MAP.items():
        if cn_word in text and random.random() < p:
            try:
                em = mood_to_emoji(en_mood)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mood_to_emoji 调用失败: %r, %s", en_mood, exc)
                continue
            if em and em != "🤷":
                text = text.replace(cn_word, cn_word + em, 1)
    return text


def emoji_strip(text: str) -> str:
    return _get_emoji_lib().replace_emoji(text, "")


def homoglyph_substitute(
    text: str, p: float = 0.05, min_similarity: float | None = None
) -> str:
    """按概率 `p` 把字替换成形近字

    默认直接信任 `_HOMOGLYPH_CANDIDATES`（已人工核定）。传入 `min_similarity` 时会
    额外用 `char_similar_z` 做形状相似度筛选，但该指标识别不了繁简/日式异体这类真
    同形字（`户→戶` 只有 0.443），除非确知用途否则不要开启
    """
    out: list[str] = []
    for ch in text:
        candidate = _HOMOGLYPH_CANDIDATES.get(ch)
        if candidate is None or random.random() >= p:
            out.append(ch)
            continue
        if min_similarity is not None:
            try:
                if _cached_shape_similarity(ch, candidate) < min_similarity:
                    out.append(ch)
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "同形字相似度计算失败: %r -> %r, %s", ch, candidate, exc
                )
                out.append(ch)
                continue
        out.append(candidate)
    return "".join(out)


def random_space(text: str, p: float = 0.04) -> str:
    """随机在字符后插入空格，模拟输入中的无效分隔"""
    return "".join(ch + " " if random.random() < p else ch for ch in text)


def homoglyph_edit_distance(text1: str, text2: str, lang: str = "zhs") -> float:
    _ensure_homoglyph_dict_loaded(lang)
    return _get_homoglyphs()["hg_distance"](text1, text2, lang)


homoglyph_distance_check = homoglyph_edit_distance


class _NlpcdaAugmenter:
    def __init__(self, nlpcda_cls, **kwargs):
        self._aug = None
        try:
            self._aug = nlpcda_cls(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("nlpcda 初始化失败: %s", exc)

    def __call__(self, text: str) -> str:
        if self._aug is None:
            return text
        try:
            results = self._aug.replace(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("nlpcda 增强失败，返回原文: %s", exc)
            return text

        if not results:
            return text
        if isinstance(results, str):
            return results
        return results[0] if results else text


def build_homophone(change_rate: float = 0.05) -> Callable[[str], str]:
    return _NlpcdaAugmenter(
        _get_nlpcda()["Homophone"],
        create_num=1,
        change_rate=change_rate,
        seed=None,
    )


def build_random_delete(change_rate: float = 0.04) -> Callable[[str], str]:
    return _NlpcdaAugmenter(
        _get_nlpcda()["RandomDeleteChar"],
        create_num=1,
        change_rate=change_rate,
        seed=None,
    )


def build_char_swap(change_rate: float = 0.04) -> Callable[[str], str]:
    return _NlpcdaAugmenter(
        _get_nlpcda()["CharPositionExchange"],
        create_num=1,
        change_rate=change_rate,
        seed=None,
    )


def build_random_entity(change_rate: float = 0.15) -> Callable[[str], str]:
    return _NlpcdaAugmenter(
        _get_nlpcda()["Randomword"],
        create_num=1,
        change_rate=change_rate,
        seed=None,
    )


def _contains_protected(text: str) -> bool:
    return any(w in text for w in _PROTECTED_WORDS)


def _wrap_with_protection(fn: Callable[[str], str]) -> Callable[[str], str]:
    def wrapped(text: str) -> str:
        if _contains_protected(text):
            return text
        return fn(text)

    return wrapped


AUGMENT_FACTORIES: dict[str, Callable[[float], Callable[[str], str]]] = {
    "emoji_replace": lambda rate: partial(emoji_replace, replace_rate=rate),
    "emoji_mood": lambda rate: partial(emoji_mood_enhance, p=rate),
    "homoglyph": lambda rate: partial(homoglyph_substitute, p=rate),
    "random_space": lambda rate: partial(random_space, p=rate),
    "homophone": lambda rate: _wrap_with_protection(build_homophone(rate)),
    "random_delete": lambda rate: _wrap_with_protection(build_random_delete(rate)),
    "char_swap": lambda rate: _wrap_with_protection(build_char_swap(rate)),
    "random_entity": lambda rate: _wrap_with_protection(build_random_entity(rate)),
}

_DEFAULT_RATES: dict[str, float] = {
    "emoji_replace": 0.6,
    "emoji_mood": 0.06,
    "homoglyph": 0.05,
    "random_space": 0.04,
    "homophone": 0.05,
    "random_delete": 0.04,
    "char_swap": 0.04,
    "random_entity": 0.15,
}


class _LazyAugmentRegistry:
    def __init__(self) -> None:
        self._cache: dict[str, Callable[[str], str]] = {}
        self._lock = threading.Lock()

    def __getitem__(self, name: str) -> Callable[[str], str]:
        if name not in AUGMENT_FACTORIES:
            raise KeyError(name)
        if name not in self._cache:
            with self._lock:
                if name not in self._cache:
                    self._cache[name] = AUGMENT_FACTORIES[name](_DEFAULT_RATES[name])
        return self._cache[name]

    def __contains__(self, name: object) -> bool:
        try:
            return name in AUGMENT_FACTORIES
        except TypeError:
            return False

    def __len__(self) -> int:
        return len(AUGMENT_FACTORIES)

    def __iter__(self):
        return iter(AUGMENT_FACTORIES)

    def keys(self):
        return AUGMENT_FACTORIES.keys()

    def values(self):
        return [self[k] for k in AUGMENT_FACTORIES]

    def items(self):
        return [(k, self[k]) for k in AUGMENT_FACTORIES]

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default


AUGMENT_REGISTRY = _LazyAugmentRegistry()


def build_augment_fn(
    kinds: list[str],
    *,
    sample_rate: float = 1.0,
    homophone_rate: Optional[float] = None,
    delete_rate: Optional[float] = None,
    swap_rate: Optional[float] = None,
    entity_rate: Optional[float] = None,
    emoji_replace_rate: Optional[float] = None,
    emoji_mood_rate: Optional[float] = None,
    homoglyph_rate: Optional[float] = None,
    random_space_rate: Optional[float] = None,
) -> Optional[Callable[[str], str]]:
    """把若干增强算子串成一条链

    Args:
        kinds: 算子名列表，取值见 `AUGMENT_FACTORIES`
        sample_rate: **被增强的样本比例**。每个样本先按该概率抽签，未抽中的样本
            原样返回；抽中的才走整条算子链。取值 1.0 表示每个样本都过一遍增强，
            0.0 表示完全不增强。注意它与各算子自身的改动概率是**相乘**关系：
            实际被改写的比例 = `sample_rate` × 算子自身的命中率
        *_rate: 覆盖对应算子的默认概率（`random_space` 默认 0.04、`emoji_replace`
            默认 0.6 等），不传则用 `_DEFAULT_RATES`
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"sample_rate 必须在 [0, 1] 之间，收到 {sample_rate!r}")
    if not kinds:
        return None

    overrides: dict[str, Optional[float]] = {
        "emoji_replace": emoji_replace_rate,
        "emoji_mood": emoji_mood_rate,
        "homoglyph": homoglyph_rate,
        "random_space": random_space_rate,
        "homophone": homophone_rate,
        "random_delete": delete_rate,
        "char_swap": swap_rate,
        "random_entity": entity_rate,
    }
    rates: dict[str, float] = dict(_DEFAULT_RATES)
    for name, value in overrides.items():
        if value is not None:
            rates[name] = value

    fns: list[tuple[str, Callable[[str], str]]] = []
    for k in kinds:
        if k not in AUGMENT_FACTORIES:
            raise ValueError(
                f"未知增强类型: {k!r}，可选: {list(AUGMENT_FACTORIES)}"
            )
        fns.append((k, AUGMENT_FACTORIES[k](rates[k])))

    def apply(text: str) -> str:
        if sample_rate < 1.0 and random.random() >= sample_rate:
            return text
        original_protected = _contains_protected(text)
        for name, fn in fns:
            if original_protected and name in _PROTECTED_AUGMENT_NAMES:
                continue
            text = fn(text)
        return text

    return apply