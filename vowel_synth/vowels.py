"""教学用元音共振峰预设。

这些数值是经典共振峰合成文献中成人发音的**教学近似值**（量级参考
Peterson & Barney 1952、Klatt 1980 的典型数据，取整到 10 Hz），
不是任何特定真人的声道测量，也不用于模拟某个真实说话人的身份。

数据以结构化 numpy 数组声明，方便按字段索引、也可直接序列化为表格。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# 字段：元音记号、ASCII 别名（用于文件名/无 Unicode 控件）、
# F1..F4 中心频率(Hz)、B1..B4 带宽(Hz)
VOWEL_DTYPE = np.dtype([
    ("symbol", "U4"),
    ("ascii", "U4"),
    ("f", np.float64, 4),
    ("bw", np.float64, 4),
])

# 带宽采用 Klatt 风格的固定教学近似（F1 宽、高频略放宽），
# 不随中心频率缩放，方便单独观察“窄/宽带宽”对峰尖锐度的影响。
_DEFAULT_BW = (80.0, 90.0, 120.0, 180.0)

# (IPA 记号, ASCII 文件名别名（全小写、无大小写冲突）, [F1..F4] Hz, 教学带宽)
_RAW_PRESETS = [
    ("i", "iy", (270, 2290, 3010, 3500), _DEFAULT_BW),  # heed
    ("ɪ", "ih", (390, 1990, 2550, 3500), _DEFAULT_BW),  # hid
    ("ɛ", "eh", (530, 1840, 2480, 3500), _DEFAULT_BW),  # head
    ("a", "aa", (730, 1090, 2440, 3500), _DEFAULT_BW),  # had
    ("ɑ", "ah", (730, 1090, 2440, 3500), _DEFAULT_BW),  # hod（与 a 近似）
    ("ɔ", "ao", (570,  840, 2410, 3500), _DEFAULT_BW),  # hawed
    ("ʊ", "uh", (440, 1020, 2240, 3500), _DEFAULT_BW),  # hood
    ("u", "uw", (300,  870, 2240, 3500), _DEFAULT_BW),  # who'd
    ("ə", "ax", (500, 1500, 2500, 3500), _DEFAULT_BW),  # schwa
]


def vowel_table() -> npt.NDArray[VOWEL_DTYPE]:
    """返回元音预设的结构化数组副本。"""
    table = np.empty(len(_RAW_PRESETS), dtype=VOWEL_DTYPE)
    for i, (sym, ascii_name, freqs, bws) in enumerate(_RAW_PRESETS):
        table[i] = (sym, ascii_name,
                    np.asarray(freqs, dtype=np.float64),
                    np.asarray(bws, dtype=np.float64))
    if len(set(table["ascii"])) != len(table):
        raise ValueError("ASCII vowel aliases must be unique")
    return table


def get_vowel(symbol: str) -> tuple[npt.NDArray[np.float64],
                                    npt.NDArray[np.float64]]:
    """按记号（IPA 或 ASCII 别名）取 (中心频率, 带宽)；找不到抛 KeyError。"""
    table = vowel_table()
    for row in table:
        if str(row["symbol"]) == symbol or str(row["ascii"]) == symbol:
            return np.array(row["f"], dtype=np.float64), \
                   np.array(row["bw"], dtype=np.float64)
    raise KeyError(f"unknown vowel {symbol!r}; "
                   f"available: {list_symbols()}")


def list_symbols() -> list[str]:
    return [str(r["symbol"]) for r in vowel_table()]


def list_ascii() -> list[str]:
    return [str(r["ascii"]) for r in vowel_table()]
