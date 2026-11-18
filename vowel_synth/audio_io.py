"""音频输入输出。

离线 WAV 只用标准库 wave，无原生依赖；
实时回放使用可选的 sounddevice（PortAudio），缺失时给出明确错误并提示
改用离线渲染，而不是假装能出声。
"""

from __future__ import annotations

import wave

import numpy as np


def write_wav(path: str, y: np.ndarray, fs: int) -> None:
    """写 16-bit PCM WAV，自动按峰值归一化到 int16 范围。"""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("只支持单声道")
    peak = np.max(np.abs(y))
    scaled = y / peak if peak > 0 else y
    pcm = np.clip(scaled * 32767.0, -32768.0, 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(pcm.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        fs = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm.astype(float) / 32768.0, fs


def sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        return False
    return True


class RealtimePlayer:
    """sounddevice 流式回放包装；构造时即检测后端与设备。"""

    def __init__(self, fs: int, block_size: int = 256):
        if not sounddevice_available():
            raise RuntimeError(
                "未找到 sounddevice/PortAudio，无法实时回放；"
                "请安装 PortAudio 后 pip install sounddevice，"
                "或使用离线 WAV 输出（render + write_wav）。")
        import sounddevice as sd
        self._sd = sd
        self.fs = fs
        self.block_size = block_size
        self.stream = None

    def start(self, callback) -> None:
        """callback(block_size)->np.float32 块，由 PortAudio 线程拉取。"""
        def _cb(outdata, frames, time_info, status):  # noqa: ANN001
            if status:
                # 欠载等状态不致命，交给回调继续供数
                pass
            block = callback(frames)
            outdata[:, 0] = block[:frames].astype(np.float32)

        self.stream = self._sd.OutputStream(
            samplerate=self.fs, blocksize=self.block_size, channels=1,
            dtype="float32", callback=_cb)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
