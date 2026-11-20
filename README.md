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
| `vowel_synth/source.py` | Rosenberg 声门流导数 / 锯齿声源 + 独立白噪送气分量；可调**源谱倾斜**（无源共振的一阶节混合）；逐样本 f0 扫描，块间相位连续 |
| `vowel_synth/filters.py` | Klatt 双极点谐振器、稳定系数设计、越界/过窄带宽钳制与告警、级联处理 |
| `vowel_synth/engine.py` | `SynthParams`、离线 `render`（支持 f0/频率/带宽/倾斜/气声/发声包络与固定 seed）、`RealtimeEngine` 低延迟块处理 |
| `vowel_synth/analysis.py` | 谐波间距估计；**白噪激励实测传递函数**的共振峰核验；LPC 粗估计 |
| `vowel_synth/audio_io.py` | 标准库 WAV 读写；可选 sounddevice/PortAudio 实时回放；`BufferPlayer` 可暂停/定位，无声卡时静音走带 |
| `vowel_synth/vowels.py` | 结构化数组声明的 9 个教学近似元音预设（附 ASCII 别名） |
| `vowel_synth/gui.py` | 源/滤后信号、频谱、共振响应、实际系数与钳制告警的交互界面 |
| `vowel_synth/exercise.py` | 练习谱面（三层参数轨迹）、安全基频、答案轨迹、音高穿带示范、跨基频可解释性核验、资产写出与图 |
| `vowel_synth/exercise_gui.py` | 乐器窗口：暂停、逐层固定、从实际输出测差异起点、隐藏答案 |
| `vowel_synth/cli.py` | 命令行：合成、扫描、自检、练习集重建/核验/示范、两个 GUI |

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

# 同音高、异声道练习集：重建音频 + 参数 + 答案轨迹
python scripts/make_exercise.py            # 或 python -m vowel_synth.cli build
python -m vowel_synth.cli build --hide-answer -o out/blind

# 更换基频重新生成，并做跨基频可解释性核验
python -m vowel_synth.cli build --f0 300 -o out/f300
python -m vowel_synth.cli verify --alt-f0 120 260 320 --bands-demo

# 音高穿过两个共振带之间的示范
python -m vowel_synth.cli demo

# 乐器窗口：暂停 / 固定某一层 / 由实际输出看差异起点
python -m vowel_synth.cli exercise-gui
```

## 同音高、异声道练习集（`exercise.py`）

练习由**可编辑的三层参数轨迹**构成，不依赖预录答案：

| 层 | 参数 | 在哪个练习中变化 |
|---|---|---|
| 声源 | f0 滑音、声门开相、**源谱倾斜** `tilt_db` | 两组共享 f0；B 组改倾斜 |
| 声道 | F1..F4 中心频率与带宽 | A 组移共振峰 |
| 气声 | 送气底噪 + 一次升余弦气声事件 | 两组逐样本相同 |

三个版本共享同一条 f0 滑音与同一次气声事件：

- `anchor`：参照；
- `a_formant`：声源逐样本不变，只有共振峰在滑音段移动；
- `b_tilt`：声道逐样本不变，只有源谱倾斜在滑音段变化。

**只靠滑音后才分辨**由实际输出量化，而不是靠计划值：`pair_a_vs_b` 报告
逐帧谱形距离在滑音前 ≈ 0 dB、滑音后 ≈ 13 dB，并用实际音频测得分岔起点
（与计划滑音段一致；改变滑音位置，起点随之移动）。

### 基线兼容与“安全基频”

- 生成时在请求值 ±12% 内搜索一个基频，使**两个稳定段**里所有谐波都离
  共振峰中心足够远（默认 ≥ 0.15·f0），把“最近的谐波”与共振峰中心记录进
  答案。滑音过程中谐波必然扫过共振峰——那正是要教的现象，不据此否定候选。
- 需要人工微调时：`--anchor/--moved/--tilt/--seed/--f0` 均可传参，
  参数写入 `spec.json`，可直接重建出逐样本相同的音频（有测试保证）。

### 乐器窗口（`exercise-gui`）

- **暂停**：Play/Pause 冻结走带；暂停后可拖动时间滑杆看任意时刻。
- **固定某一层**：勾选 `Fix source.tilt / tract.formants / air.breath`，
  该层冻结在滑音前取值，其余照常演奏。
- **差异来自实际输出**：窗口对每个版本即时渲染“当前设置”与“只冻结自身
  变化层”的参照，并测量两者的谱形距离。固定声道层后 A 的曲线归零；
  固定源谱层后 B 的曲线归零——归因干净，可逐层演示。
- 无声卡环境自动降级为**静音走带**并在状态栏注明，不假装出声；可用
  `--hide-answer` 或按钮隐藏参数轨迹，实测谱形与差异曲线始终保留。

### 音高穿带示范

`PitchDemoSpec` 让 f0 从 380 Hz 扫到 1100 Hz，依次穿过 F1≈520 Hz 与
F2≈980 Hz。声道包络峰始终固定在两个设计值；但 f0≈753 Hz 时基波正好落在
两带之间且是**最强谐波**。答案显式记录“最强谐波到各共振峰中心的偏移”，
用来提醒：频谱上的局部最强点不等于声道共振中心。

### 跨基频可解释性核验

`cross_pitch_check(built, alt_f0)` 用**另一基频**重新渲染实际输出，然后：

1. 对滑音后的若干帧，预测 `输出谐波 = 源谱(含倾斜) + 声道包络`；
2. 与实测谐波比较（默认根均方 < 3 dB、相关 > 0.9）；
3. 报告来自参数的**包络峰位**（不随 f0 改变）与“就近取谐波”的朴素估计
   （随 f0 移动），说明为何不能用某个谐波冒充共振峰。

`python -m vowel_synth.cli verify --alt-f0 120 260 320` 会打印逐帧结果；
`scripts/make_exercise.py` 还会写 `cross_pitch_checks.json`。

## 如何验证参数关系

1. **谐波间距 = f0**：`estimate_harmonic_spacing` 从有声谱峰间距反推基频，
   典型误差 < 0.3 Hz。
2. **共振峰 = 传递函数极点**：用平坦白噪激励同一级联，Welch 平均测 `H(f)`，
   分离良好的共振峰实测峰位与设计值偏差 < 5%（多数 < 1%）。
   不用有声信号直接找峰，因为声门谱倾斜和谐波栅格会污染估计。
3. **音高/共振峰解耦**：f0 改变前后两次 `render` 的 `effective` 系数逐元素相同；
   实测传递函数最大变化为 0 dB。
4. **源谱倾斜不造峰**：`SpectralTilt.response_db` 在整个频带单调下降，
   且倾斜改变前后共振峰系数完全相同（`test_tilt_does_not_move_formants`）。
5. **带宽控制峰尖锐度**：`verification.png` 右下对比 0.5x/1x/2x 带宽。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

40 项测试覆盖：滤波器极点稳定、钳制边界、块状态连续、声源相位连续与扫描、
解耦不变量、谐波间距、白噪法共振峰、扫描连续性、WAV 往返；以及练习集的
三层共享/单层变化、滑音前接近滑音后拉开、实测分岔起点随滑音计划移动、
安全基频的谐波间距、换基频重建、跨基频可解释性、隐藏答案不写答案文件、
`spec.json` 逐样本重建、乐器窗口的逐层归因与静音走带。
