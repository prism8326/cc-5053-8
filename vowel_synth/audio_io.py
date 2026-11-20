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


class BufferPlayer:
    """播放一段固定缓冲，可暂停/续播，并暴露播放位置。

    有声卡时走 PortAudio 回调；没有声卡时仍可“走带”（cursor 前进），
    由调用方在界面上明确标注为静音预览，绝不假装在出声。
    """

    def __init__(self, y: np.ndarray, fs: int, block_size: int = 512):
        self.fs = fs
        self.block_size = block_size
        self._y = np.asarray(y, dtype=np.float32)
        self.pos = 0               # 已消费样本数
        self.playing = False
        self.stream = None
        self.audible = False

    @property
    def duration(self) -> float:
        return self._y.size / self.fs

    @property
    def time(self) -> float:
        return self.pos / self.fs

    def load(self, y: np.ndarray) -> None:
        self.stop()
        self._y = np.asarray(y, dtype=np.float32)
        self.pos = 0

    def play(self) -> None:
        if self.playing:
            return
        if self.pos >= self._y.size:
            self.pos = 0
        self.playing = True
        if sounddevice_available():
            import sounddevice as sd

            def _cb(outdata, frames, time_info, status):  # noqa: ANN001
                n = min(frames, self._y.size - self.pos)
                if n > 0:
                    outdata[:n, 0] = self._y[self.pos:self.pos + n]
                    self.pos += n
                if n < frames:
                    outdata[n:, 0] = 0.0
                if self.pos >= self._y.size:
                    self.playing = False
                    raise sd.CallbackStop

            self.stream = self._sd_out(sd, _cb)
            self.stream.start()
            self.audible = True

    def _sd_out(self, sd, callback):  # noqa: ANN001
        return sd.OutputStream(samplerate=self.fs, blocksize=self.block_size,
                               channels=1, dtype="float32",
                               callback=callback)

    def advance(self, seconds: float) -> None:
        """离线模式下由界面定时器推进走带位置。"""
        if self.playing:
            self.pos = min(self._y.size,
                           self.pos + int(round(seconds * self.fs)))
            if self.pos >= self._y.size:
                self.playing = False

    def pause(self) -> None:
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        self.audible = False

    def stop(self) -> None:
        self.pause()
        self.pos = 0
