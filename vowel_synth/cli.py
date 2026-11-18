"""命令行入口：离线渲染、音高扫描、频谱自检与 GUI。

示例：
  python -m vowel_synth.cli synth --vowel a --f0 120 -o a.wav
  python -m vowel_synth.cli sweep --f0-start 80 --f0-end 400 -o sweep.wav
  python -m vowel_synth.cli selfcheck --vowel i --f0 100
  python -m vowel_synth.cli gui
"""

from __future__ import annotations

import argparse

import numpy as np

from .analysis import (estimate_harmonic_spacing,
                       measure_transfer_function, verify_formants)
from .audio_io import sounddevice_available, write_wav
from .engine import SynthParams, render
from .vowels import get_vowel, list_ascii, list_symbols, vowel_table


def _params_from_args(a: argparse.Namespace) -> SynthParams:
    freqs, bws = get_vowel(a.vowel)
    if a.formant_freq is not None:
        for spec in a.formant_freq:
            k, f = spec.split("=")
            freqs[int(k) - 1] = float(f)
    if a.formant_bw is not None:
        for spec in a.formant_bw:
            k, bw = spec.split("=")
            bws[int(k) - 1] = float(bw)
    return SynthParams(f0=a.f0, freqs=freqs, bws=bws,
                       open_quotient=a.open_quotient, shape=a.shape,
                       noise_gain=a.noise)


def cmd_synth(a: argparse.Namespace) -> None:
    result = render(a.duration, a.fs, _params_from_args(a), block_size=a.block)
    write_wav(a.output, result["y"], a.fs)
    print(f"已写 {a.output}：{a.duration}s @ {a.fs}Hz，"
          f"f0={a.f0}Hz，元音 /{a.vowel}/")
    _print_unique_warnings(result)


def _print_unique_warnings(result: dict) -> None:
    seen = set()
    for block_warn in result["warnings"]:
        for msg in block_warn["messages"]:
            if msg not in seen:
                seen.add(msg)
                print("  钳制告警:", msg)


def cmd_sweep(a: argparse.Namespace) -> None:
    p = _params_from_args(a)
    n = int(round(a.duration * a.fs))
    n_blocks = int(np.ceil(n / a.block))
    f0_env = np.linspace(a.f0_start, a.f0_end, n_blocks)
    result = render(a.duration, a.fs, p, block_size=a.block,
                    f0_envelope=f0_env)
    write_wav(a.output, result["y"], a.fs)
    # 验证扫描期间共振峰系数恒定（只有 f0 在动）
    eff = result["effective"]
    f_drift = np.abs(eff[:, :, 0] - eff[0, :, 0]).max()
    print(f"扫描 {a.f0_start}→{a.f0_end}Hz 已写 {a.output}")
    print(f"扫描期间共振峰生效频率最大漂移 = {f_drift:.2e} Hz（应≈0，"
          f"证明音高与共振峰解耦）")
    _print_unique_warnings(result)


def cmd_selfcheck(a: argparse.Namespace) -> None:
    p = _params_from_args(a)
    result = render(0.8, a.fs, p, block_size=a.block)
    f0_est = estimate_harmonic_spacing(result["y"], a.fs)
    print(f"[谐波间距] 设定 f0 = {p.f0:7.2f} Hz | 谱面估计 = {f0_est:7.2f} Hz "
          f"（误差 {abs(f0_est - p.f0):.2f} Hz）")

    print("[共振包络] 白噪激励实测声道传递函数：")
    for r in verify_formants(p.freqs, p.bws, a.fs):
        flag = "OK" if r["ok"] else "合并/超差"
        print(f"  目标 {r['target']:7.1f} Hz | 实测 {r['measured']:7.1f} Hz "
              f"| 相对误差 {r['rel_error']*100:5.2f}%  {flag}")

    # 解耦验证：f0 改变后，声道传递函数必须保持不变
    f1, H1 = measure_transfer_function(p.freqs, p.bws, a.fs, duration=2.0)
    p2 = SynthParams(f0=p.f0 * 1.5, freqs=p.freqs.copy(), bws=p.bws.copy(),
                     open_quotient=p.open_quotient, shape=p.shape,
                     noise_gain=p.noise_gain)
    f0_est2 = estimate_harmonic_spacing(
        render(0.8, a.fs, p2, block_size=a.block)["y"], a.fs)
    f2, H2 = measure_transfer_function(p2.freqs, p2.bws, a.fs, duration=2.0)
    h_diff = np.max(np.abs(H1 - H2))
    print(f"[解耦检查] f0 {p.f0:.0f}→{p2.f0:.0f}Hz（谱面估计 {f0_est2:.1f}Hz），"
          f"传递函数最大变化 = {h_diff:.2e} dB（应≈0）")


def cmd_vowels(_a: argparse.Namespace) -> None:
    t = vowel_table()
    print("教学近似元音预设（非任何真人数据）：")
    print(f"{'IPA':<5}{'别名':<5}{'F1':>6}{'F2':>7}{'F3':>7}{'F4':>7}"
          f"{'B1':>6}{'B2':>6}{'B3':>6}{'B4':>6}")
    for row in t:
        print(f"{str(row['symbol']):<5}{str(row['ascii']):<5}" +
              "".join(f"{v:7.0f}" for v in row["f"]) +
              "".join(f"{v:6.0f}" for v in row["bw"]))
    print("IPA 记号:", " ".join(list_symbols()))
    print("ASCII 别名:", " ".join(list_ascii()))


def cmd_gui(_a: argparse.Namespace) -> None:
    from .gui import VowelSynthGUI
    gui = VowelSynthGUI()
    if not sounddevice_available():
        print("提示：未检测到 PortAudio/sounddevice，GUI 运行于离线模式，"
              "可用按钮导出 WAV。")
    gui.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vowel_synth",
        description="本地共振峰元音合成：可控周期声源+噪声激励共振峰级联")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--vowel", default="a",
                        choices=list_symbols() + list_ascii(),
                        help="IPA 记号或 ASCII 别名（见 vowels 子命令）")
        sp.add_argument("--f0", type=float, default=120.0)
        sp.add_argument("--open-quotient", type=float, default=0.6)
        sp.add_argument("--shape", choices=["rosenberg", "saw"],
                        default="rosenberg")
        sp.add_argument("--noise", type=float, default=0.02)
        sp.add_argument("--formant-freq", action="append",
                        help="覆盖共振峰频率，如 1=800 （F1=800Hz），可重复")
        sp.add_argument("--formant-bw", action="append",
                        help="覆盖共振峰带宽，如 2=150，可重复")
        sp.add_argument("--fs", type=int, default=22050)
        sp.add_argument("--block", type=int, default=512)

    sp = sub.add_parser("synth", help="离线渲染一个元音到 WAV")
    add_common(sp)
    sp.add_argument("-d", "--duration", type=float, default=1.5)
    sp.add_argument("-o", "--output", default="demo_output/vowel.wav")
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("sweep", help="连续音高扫描（共振峰保持恒定）")
    add_common(sp)
    sp.add_argument("--f0-start", type=float, default=80.0)
    sp.add_argument("--f0-end", type=float, default=400.0)
    sp.add_argument("-d", "--duration", type=float, default=1.5)
    sp.add_argument("-o", "--output", default="demo_output/pitch_sweep.wav")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("selfcheck", help="频谱自检：谐波间距与共振包络")
    add_common(sp)
    sp.set_defaults(func=cmd_selfcheck)

    sp = sub.add_parser("vowels", help="列出教学元音预设")
    sp.set_defaults(func=cmd_vowels)

    sp = sub.add_parser("gui", help="启动交互界面")
    sp.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
