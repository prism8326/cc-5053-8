"""共振峰滤波器的稳定性与边界处理测试。"""

import numpy as np
import pytest

from vowel_synth.filters import (F_MIN, NYQUIST_MARGIN, R_MAX,
                                 FormantCascade, design_resonator)


def test_resonator_poles_inside_unit_circle():
    for fs in (8000, 22050, 44100):
        for f in np.linspace(50, fs * 0.49, 20):
            for bw in (20.0, 80.0, 300.0):
                spec = design_resonator(f, bw, fs)
                assert spec.stable
                assert abs(spec.a2) < 1.0
                # 极点半径直接核算
                R = np.sqrt(abs(spec.a2))
                assert R <= R_MAX + 1e-12


def test_narrow_bandwidth_is_widened_with_warning():
    fs = 22050
    spec = design_resonator(1000.0, 1.0, fs)
    bw_min = -fs / np.pi * np.log(R_MAX)
    assert spec.bw_effective == pytest.approx(bw_min, rel=1e-9)
    assert spec.bw_requested == 1.0
    assert any("too narrow" in w for w in spec.warnings)
    assert spec.stable


def test_frequency_above_nyquist_clamped():
    fs = 22050
    spec = design_resonator(fs * 0.6, 100.0, fs)
    f_hi = 0.5 * fs * (1 - NYQUIST_MARGIN)
    assert spec.f_effective == pytest.approx(f_hi)
    assert any("Nyquist" in w for w in spec.warnings)


def test_frequency_below_floor_clamped():
    spec = design_resonator(0.5, 100.0, 22050)
    assert spec.f_effective == F_MIN
    assert any("below floor" in w for w in spec.warnings)


def test_nonpositive_bandwidth_handled():
    spec = design_resonator(1000.0, 0.0, 22050)
    assert spec.stable and spec.bw_effective > 0
    assert spec.warnings


def test_cascade_preserves_state_and_stays_finite():
    fs = 22050
    cas = FormantCascade(4, fs)
    rng = np.random.default_rng(1)
    freqs = np.array([730, 1090, 2440, 3500.])
    bws = np.array([80, 90, 120, 180.])
    for _ in range(20):
        x = rng.standard_normal(512)
        y, warns = cas.process_block(x, freqs, bws)
        assert np.all(np.isfinite(y))
        assert all(s.stable for s in cas.specs)


def test_cascade_rejects_wrong_param_count():
    cas = FormantCascade(3, 22050)
    with pytest.raises(ValueError):
        cas.process_block(np.zeros(64), np.array([1, 2, 3, 4.]),
                          np.array([1, 2, 3, 4.]))


def test_frequency_response_peak_near_design_frequency():
    fs = 22050
    cas = FormantCascade(1, fs)
    f0 = 1000.0
    fr, db = cas.frequency_response(np.array([f0]), np.array([60.]), n_fft=8192)
    peak_f = fr[np.argmax(db)]
    assert abs(peak_f - f0) < 30.0  # FFT 栅格 + 峰宽范围内
