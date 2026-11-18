"""Matplotlib interactive interface.

Plots: source signal, filtered signal (time + spectrum), formant cascade
frequency response, and the actual effective coefficients.
Controls: pitch f0, glottal shape (open quotient / waveform / aspiration),
and per-formant center frequency and bandwidth.

Pitch (f0) drives only the source phase accumulator; the formant sliders
control the vocal-tract cascade independently. When a sound card is present
the GUI plays through a low-latency PortAudio stream; otherwise it runs in
offline mode (every slider movement re-renders locally) and can export WAV.
The synthesis model is identical in both modes.
"""

from __future__ import annotations

import os

import matplotlib

# Headless containers fall back to Agg so the window can still be built and
# saved to an image; desktop environments keep their default interactive backend.
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.widgets import Button, RadioButtons, Slider  # noqa: E402

from .analysis import magnitude_spectrum  # noqa: E402
from .audio_io import (RealtimePlayer, sounddevice_available,  # noqa: E402
                       write_wav)
from .engine import RealtimeEngine, SynthParams, render  # noqa: E402
from .filters import FormantCascade  # noqa: E402
from .vowels import get_vowel, list_ascii, list_symbols  # noqa: E402

FS = 22050
BLOCK = 512
N_FORMANTS = 4
PREVIEW_SECONDS = 0.08


class VowelSynthGUI:
    def __init__(self, fs: int = FS):
        self.fs = fs
        self.f0 = 120.0
        self.open_quotient = 0.6
        self.noise_gain = 0.02
        self.shape = "rosenberg"
        freqs, bws = get_vowel("a")
        self.freqs, self.bws = freqs.copy(), bws.copy()
        self.player = None
        self.engine = None
        self.timer = None
        self.realtime = sounddevice_available()
        self._status_cache = ""
        self._build_figure()
        self._build_widgets()
        if self.realtime:
            self._start_realtime()
        else:
            self._status("No audio device: offline mode "
                         "(sliders re-render locally; export WAV to listen).")
        self.refresh()

    # ---------------- layout ----------------
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 11))
        self.fig.suptitle("Formant Vowel Synthesizer  "
                          "(glottal source and vocal tract decoupled)",
                          fontsize=14)
        # Top 62%: plots. Bottom 38%: controls and coefficient readout.
        gs = self.fig.add_gridspec(
            2, 2, height_ratios=[1, 1],
            left=0.06, right=0.97, top=0.93, bottom=0.47,
            hspace=0.45, wspace=0.22)
        self.ax_src = self.fig.add_subplot(gs[0, 0])
        self.ax_out = self.fig.add_subplot(gs[0, 1])
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.ax_resp = self.fig.add_subplot(gs[1, 1])
        self.ax_coef = self.fig.add_axes([0.06, 0.335, 0.91, 0.085])
        self.ax_coef.axis("off")
        self.ax_status = self.fig.add_axes([0.82, 0.012, 0.16, 0.11])
        self.ax_status.axis("off")

    def _build_widgets(self) -> None:
        self.sliders: dict[str, Slider] = {}
        h, gap = 0.018, 0.013
        top = 0.325
        col_x, col_w = (0.07, 0.20), (0.40, 0.20)

        def add_slider(col: int, row: int, key: str, label: str,
                       lo: float, hi: float, val: float) -> None:
            x, w = col_x[col], col_w[col]
            y = top - row * (h + gap)
            ax = self.fig.add_axes([x, y, w, h])
            s = Slider(ax, label, lo, hi, valinit=val, initcolor="none")
            s.label.set_position((0.0, 1.35))
            s.label.set_fontsize(8)
            s.valtext.set_fontsize(8)
            s.on_changed(self._slider_event(key))
            self.sliders[key] = s

        # Left column: source parameters.
        add_slider(0, 0, "f0", "Pitch f0 (Hz)", 60, 500, self.f0)
        add_slider(0, 1, "oq", "Glottal open quotient", 0.1, 0.95,
                   self.open_quotient)
        add_slider(0, 2, "noise", "Aspiration noise", 0.0, 0.3,
                   self.noise_gain)
        # Right column: four formants, frequency/width pairs.
        f_hi = 0.5 * self.fs * 0.95
        for k in range(N_FORMANTS):
            add_slider(1, 2 * k, f"f{k+1}", f"F{k+1} f (Hz)",
                       50, f_hi, self.freqs[k])
            add_slider(1, 2 * k + 1, f"b{k+1}", f"F{k+1} bw (Hz)",
                       10, 600, self.bws[k])

        # Preset selectors on their own row below the sliders.
        # ASCII display labels map by position onto the IPA preset symbols.
        gui_labels = list_ascii()
        self.vowel_ax = self.fig.add_axes([0.07, 0.01, 0.10, 0.085])
        self.vowel_radio = RadioButtons(
            self.vowel_ax, gui_labels,
            active=list_symbols().index("a"))
        self.vowel_radio.on_clicked(
            lambda _lbl: self._on_vowel(list_symbols()[
                gui_labels.index(self.vowel_radio.value_selected)]))
        self.vowel_ax.set_title("vowel", fontsize=8)
        for lbl in self.vowel_radio.labels:
            lbl.set_fontsize(7)
        self.shape_ax = self.fig.add_axes([0.18, 0.02, 0.10, 0.06])
        self.shape_radio = RadioButtons(
            self.shape_ax, ("rosenberg", "saw"), active=0)
        self.shape_radio.on_clicked(self._on_shape)
        self.shape_ax.set_title("shape", fontsize=8)
        for lbl in self.shape_radio.labels:
            lbl.set_fontsize(7)

        buttons = [
            ("Play / Stop", self._toggle_play),
            ("Pitch sweep demo", self._pitch_sweep),
            ("Export WAV (1.5s)", self._export_wav),
            ("Spectrum self-check", self._spectrum_selfcheck),
            ("Reset", self._reset),
        ]
        self.buttons = []
        for i, (label, cb) in enumerate(buttons):
            axb = self.fig.add_axes([0.67, 0.31 - i * 0.028, 0.135, 0.022])
            b = Button(axb, label)
            b.on_clicked(cb)
            self.buttons.append(b)
        self.status_txt = self.ax_status.text(
            0.0, 1.0, "", va="top", fontsize=8, wrap=True)

    # ---------------- parameter plumbing ----------------
    def _slider_event(self, key: str):
        def handler(_value: float) -> None:
            if key == "f0":
                self.f0 = self.sliders["f0"].val
            elif key == "oq":
                self.open_quotient = self.sliders["oq"].val
            elif key == "noise":
                self.noise_gain = self.sliders["noise"].val
            elif key[0] == "f":
                self.freqs[int(key[1:]) - 1] = self.sliders[key].val
            elif key[0] == "b":
                self.bws[int(key[1:]) - 1] = self.sliders[key].val
            self._on_slider()
        return handler

    def _current_params(self) -> SynthParams:
        return SynthParams(f0=self.f0, freqs=self.freqs.copy(),
                           bws=self.bws.copy(),
                           open_quotient=self.open_quotient,
                           shape=self.shape, noise_gain=self.noise_gain)

    def _on_slider(self) -> None:
        if self.engine is not None:
            self.engine.update_params(self._current_params())
        self.refresh()

    def _on_vowel(self, label: str) -> None:
        f, b = get_vowel(label)
        self.freqs, self.bws = f.copy(), b.copy()
        for k in range(N_FORMANTS):
            self.sliders[f"f{k+1}"].set_val(self.freqs[k])
            self.sliders[f"b{k+1}"].set_val(self.bws[k])
        self.refresh()

    def _on_shape(self, label: str) -> None:
        self.shape = label
        if self.engine is not None:
            self.engine.update_params(self._current_params())
        self.refresh()

    def _reset(self, _event=None) -> None:
        self.f0, self.open_quotient, self.noise_gain = 120.0, 0.6, 0.02
        self.freqs, self.bws = get_vowel("a")
        for key, val in [("f0", self.f0), ("oq", self.open_quotient),
                         ("noise", self.noise_gain)]:
            self.sliders[key].set_val(val)
        for k in range(N_FORMANTS):
            self.sliders[f"f{k+1}"].set_val(self.freqs[k])
            self.sliders[f"b{k+1}"].set_val(self.bws[k])
        self.refresh()

    # ---------------- audio ----------------
    def _start_realtime(self) -> None:
        self.engine = RealtimeEngine(self.fs, self._current_params(), BLOCK)
        self.player = RealtimePlayer(self.fs, BLOCK)
        self.player.start(lambda n: self.engine.process_block(n))
        self._status("Live playback running (PortAudio).")
        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self.refresh)
        self.timer.start()

    def _toggle_play(self, _event=None) -> None:
        if not sounddevice_available():
            self._status("No audio backend: use Export WAV to listen.")
            return
        if self.player is not None:
            self.player.stop()
            self.player = None
            if self.timer is not None:
                self.timer.stop()
            self._status("Stopped.")
        else:
            self._start_realtime()

    def _pitch_sweep(self, _event=None) -> None:
        duration = 1.5
        n = int(duration * self.fs)
        n_blocks = int(np.ceil(n / BLOCK))
        f0_env = np.linspace(80.0, 400.0, n_blocks)
        # Only f0 changes; formant coefficients stay constant by construction.
        result = render(duration, self.fs, self._current_params(),
                        block_size=BLOCK, f0_envelope=f0_env)
        path = "demo_output/pitch_sweep.wav"
        write_wav(path, result["y"], self.fs)
        msg = (f"Sweep {f0_env[0]:.0f}->{f0_env[-1]:.0f} Hz with fixed "
               f"formants; saved {path}")
        if result["warnings"]:
            msg += "; clamp: " + "; ".join(result["warnings"][0]["messages"])
        self._status(msg)
        self._plot_result(result, title="Pitch sweep (formants held constant)")

    def _export_wav(self, _event=None) -> None:
        result = render(1.5, self.fs, self._current_params(), BLOCK)
        path = "demo_output/vowel.wav"
        write_wav(path, result["y"], self.fs)
        self._status(f"Exported {path}")
        self._plot_result(result)

    def _spectrum_selfcheck(self, _event=None) -> None:
        from .analysis import (estimate_harmonic_spacing,
                               verify_formants)
        result = render(0.8, self.fs, self._current_params(), BLOCK)
        f0_est = estimate_harmonic_spacing(result["y"], self.fs)
        lines = [f"Set f0={self.f0:.1f} Hz, measured harmonic spacing="
                 f"{f0_est:.1f} Hz"]
        for r in verify_formants(self.freqs, self.bws, self.fs):
            lines.append(
                f"target {r['target']:6.1f} Hz / measured {r['measured']:6.1f}"
                f" Hz ({r['rel_error']*100:4.1f}%)")
        self._status("\n".join(lines))

    # ---------------- plotting ----------------
    def _preview_render(self) -> dict:
        return render(PREVIEW_SECONDS, self.fs, self._current_params(), BLOCK,
                      normalize=False)

    def refresh(self) -> None:
        result = self._preview_render()
        self._plot_result(result)
        self._show_coefficients(result)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _plot_result(self, result: dict, title: str | None = None) -> None:
        t = np.arange(result["y"].size) / self.fs
        xlim = (0, min(0.05, t[-1]))

        self.ax_src.clear()
        self.ax_src.plot(t, result["source"], lw=0.7)
        self.ax_src.set_title("Source: glottal excitation + aspiration")
        self.ax_src.set_xlabel("time (s)")
        self.ax_src.set_xlim(xlim)

        self.ax_out.clear()
        self.ax_out.plot(t, result["y"], lw=0.7, color="tab:orange")
        self.ax_out.set_title("Filtered: formant cascade output")
        self.ax_out.set_xlabel("time (s)")
        self.ax_out.set_xlim(xlim)

        self.ax_spec.clear()
        for sig, name, c in [(result["source"], "source", "tab:blue"),
                             (result["y"], "output", "tab:orange")]:
            fr, mag = magnitude_spectrum(sig, self.fs)
            band = fr <= 5000
            self.ax_spec.plot(fr[band], 20 * np.log10(mag[band] + 1e-9),
                              label=name, lw=0.9, color=c)
        for ff in self.freqs:
            self.ax_spec.axvline(ff, color="tab:green", ls="--", lw=0.7)
        self.ax_spec.set_title("Spectrum: harmonic spacing vs formant envelope")
        self.ax_spec.set_xlabel("frequency (Hz)")
        self.ax_spec.set_ylabel("magnitude (dB)")
        self.ax_spec.legend(fontsize=8)

        self.ax_resp.clear()
        cascade = FormantCascade(N_FORMANTS, self.fs)
        fr, db = cascade.frequency_response(self.freqs, self.bws)
        band = fr <= 5000
        self.ax_resp.plot(fr[band], db[band], color="tab:purple")
        for ff in self.freqs:
            self.ax_resp.axvline(ff, color="tab:green", ls="--", lw=0.7)
        self.ax_resp.set_title("Frequency response from actual coefficients")
        self.ax_resp.set_xlabel("frequency (Hz)")
        self.ax_resp.set_ylabel("gain (dB)")
        if title:
            self.fig.suptitle(title, fontsize=13)

    def _show_coefficients(self, result: dict) -> None:
        self.ax_coef.clear()
        self.ax_coef.axis("off")
        header = "Effective resonator coefficients (requested -> clamped; " \
                 "all poles |a2|<1)"
        lines = [header]
        half = (N_FORMANTS + 1) // 2
        for r in range(half):
            cells = []
            for k in (r, r + half):
                if k >= N_FORMANTS:
                    continue
                f_eff, b_eff, C, a1, a2 = result["effective"][-1][k]
                cells.append(
                    f"F{k+1}: f {self.freqs[k]:6.1f}->{f_eff:6.1f} | "
                    f"bw {self.bws[k]:5.1f}->{b_eff:5.1f} | "
                    f"C={C:.4f} a1={a1: .3f} a2={a2: .3f}")
            lines.append("        ".join(cells))
        if result["warnings"]:
            lines.append("CLAMP: " + " ; ".join(result["warnings"][0]["messages"]))
        self.ax_coef.text(0.0, 0.95, "\n".join(lines), va="top",
                          family="monospace", fontsize=8.5)
        self.status_txt.set_text(self._status_cache)

    def _status(self, msg: str) -> None:
        self._status_cache = msg
        if hasattr(self, "status_txt"):
            self.status_txt.set_text(msg)

    def show(self) -> None:
        plt.show()


def main() -> None:
    VowelSynthGUI().show()


if __name__ == "__main__":
    main()
