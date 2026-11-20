"""从参数（而非预录答案）重建整套练习资产。

运行：.venv/bin/python scripts/make_exercise.py [--out DIR] [--f0 HZ]
产物：WAV 对比音、spec.json、answer.json、概览图、音高穿带示范。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vowel_synth.exercise import (ExerciseSpec, build_exercise,  # noqa: E402
                                  build_pitch_demo, cross_pitch_check,
                                  save_assets, save_pitch_demo, PitchDemoSpec)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "demo_output",
                                                  "exercise"))
    ap.add_argument("--f0", type=float, default=None)
    ap.add_argument("--alt-f0", type=float, nargs="+",
                    default=[120.0, 260.0, 320.0])
    args = ap.parse_args()

    spec = ExerciseSpec()
    built = build_exercise(spec, f0=args.f0)
    files = save_assets(built, args.out)
    print(f"练习已重建 → {args.out}")
    print(f"  f0 {built.f0_info['f0_requested']:.1f} → "
          f"{built.f0_info['f0_used']:.2f} Hz；谐波离共振峰最小 "
          f"{built.f0_info['worst_harmonic_offset_hz']:.1f} Hz "
          f"(安全={built.f0_info['safe']})")

    checks = {}
    for alt in args.alt_f0:
        cp = cross_pitch_check(built, alt)
        checks[f"{alt:g}"] = cp
        print(f"  跨基频 {alt:6.1f}→{cp['alt_f0_used_hz']:7.2f} Hz："
              f"可解释={cp['ok']}，包络峰 {cp['envelope_peaks_hz']}")

    checks_path = os.path.join(args.out, "cross_pitch_checks.json")
    with open(checks_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    files["cross_pitch_checks"] = checks_path

    demo = build_pitch_demo(PitchDemoSpec())
    demo_files = save_pitch_demo(demo, args.out)
    print("  音高穿带：" + demo["finding"])
    print("  产物：", ", ".join(
        os.path.relpath(p, ROOT) for p in list(files.values())
        + list(demo_files.values())))


if __name__ == "__main__":
    main()
