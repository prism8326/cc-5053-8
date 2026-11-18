# 共振峰元音合成演示器

一个完全本地运行的教学级元音合成器：**可控周期声源 + 噪声分量**激励
**共振峰（双极点谐振器）级联**，真实合成可听音频。不调用任何语音合成服务，
也不模拟任何真实说话人的身份——只演示可控制的声音结构与可验证的参数关系。

## 核心设计：声源与声道解耦

```
f0 / 开相比 / 波形 / 送气噪声        共振峰频率 F1..F4 / 带宽 B1..B4
        │                                          │
 GlottalSource（相位累加）              FormantCascade（谐振器级联）
        └──────────► 激励信号 x ──────────────────► 滤波输出 y
```

- **基频 f0 只驱动声源的相位累加器**，决定谐波间距，绝不修改共振峰系数。
- **共振峰频率只由声道配置决定**。改变音高时谐波整体平移，共振包络保持不动。
  测试 `test_pitch_does_not_move_formants` 与音高扫描实验都验证了
  扫描期间共振峰生效频率漂移严格为 0。

## 模块

| 文件 | 职责 |
|---|---|
| `vowel_synth/source.py` | Rosenberg 声门流导数 / 锯齿声源 + 独立白噪送气分量；逐样本 f0 扫描，块间相位连续 |
| `vowel_synth/filters.py` | Klatt 双极点谐振器、稳定系数设计、越界/过窄带宽钳制与告警、级联处理 |
| `vowel_synth/engine.py` | `SynthParams`、离线 `render`（支持 f0/频率/带宽包络）、`RealtimeEngine` 低延迟块处理 |
| `vowel_synth/analysis.py` | 谐波间距估计；**白噪激励实测传递函数**的共振峰核验；LPC 粗估计 |
| `vowel_synth/audio_io.py` | 标准库 WAV 读写；可选 sounddevice/PortAudio 实时回放（缺失时显式降级） |
| `vowel_synth/vowels.py` | 结构化数组声明的 9 个教学近似元音预设（附 ASCII 别名） |
| `vowel_synth/gui.py` | 源/滤后信号、频谱、共振响应、实际系数与钳制告警的交互界面 |
| `vowel_synth/cli.py` | 命令行：合成、扫描、频谱自检、预设表、GUI |

## 稳定性与边界处理

谐振器系数 `y[n] = C·x[n] + a1·y[n-1] + a2·y[n-2]`，极点半径
`R = exp(-π·bw/fs) < 1`，在支持范围内解析保证稳定。越界不静默、不崩溃：

- 频率低于下限或逼近 Nyquist → 钳制到 `[F_MIN, fs/2·(1-2%)]`，记录英文告警；
- 带宽过窄导致 `R > 0.999` → 展宽到最小带宽 `bw_min = -fs/π·ln(0.999)`；
- 非正/非有限带宽同样钳制并告警。界面显示“请求值 → 实际生效值”和全部系数。

## 安装

```bash
python -m venv .venv
.venv/bin/pip install numpy scipy matplotlib sounddevice pytest
```

`sounddevice` 仅实时回放需要；容器无声卡时自动进入离线模式，用 WAV 试听。

## 使用

```bash
# 离线合成（IPA 记号或 ASCII 别名均可）
python -m vowel_synth.cli synth --vowel iy --f0 110 -o i.wav
python -m vowel_synth.cli synth --vowel a --formant-freq 1=800 --formant-bw 2=150

# 连续音高扫描（共振峰恒定，报告漂移量）
python -m vowel_synth.cli sweep --f0-start 80 --f0-end 400

# 频谱自检：谐波间距、白噪法共振峰核验、解耦检查
python -m vowel_synth.cli selfcheck --vowel uw --f0 130

# 列预设 / 启动 GUI
python -m vowel_synth.cli vowels
python -m vowel_synth.cli gui

# 生成全部演示 WAV 与验证图（demo_output/）
python scripts/make_demo.py
```

## 如何验证参数关系

1. **谐波间距 = f0**：`estimate_harmonic_spacing` 从有声谱峰间距反推基频，
   典型误差 < 0.3 Hz。
2. **共振峰 = 传递函数极点**：用平坦白噪激励同一级联，Welch 平均测 `H(f)`，
   分离良好的共振峰实测峰位与设计值偏差 < 5%（多数 < 1%）。
   不用有声信号直接找峰，因为声门谱倾斜和谐波栅格会污染估计。
3. **音高/共振峰解耦**：f0 改变前后两次 `render` 的 `effective` 系数逐元素相同；
   实测传递函数最大变化为 0 dB。
4. **带宽控制峰尖锐度**：`verification.png` 右下对比 0.5x/1x/2x 带宽。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

23 项测试覆盖：滤波器极点稳定、钳制边界、块状态连续、声源相位连续与扫描、
解耦不变量、谐波间距、白噪法共振峰、扫描连续性、WAV 往返。
