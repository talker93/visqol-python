# ViSQOL Python 版实施计划

## 一、项目概述

将 ViSQOL v3.3.3 (C++) 完整转换为纯 Python 实现，消除 Bazel、Armadillo、PFFFT、Abseil、TFLite C++ 等编译型依赖，仅依赖 `numpy`、`scipy`、`soundfile` 和 `libsvm`（均为 pip 可安装包）。

### 决策记录

| 决策项 | 选择 |
|--------|------|
| 运行模式 | Audio + Speech 两种模式全部支持 |
| Speech 质量映射 | 仅多项式/指数映射（不引入 TFLite 依赖） |
| Audio 质量映射 | SVR（通过 libsvm Python 绑定加载原始模型文件） |
| 数值一致性 | 实用一致（MOS 差异 < 0.05） |
| 输出位置 | `visqol/visqol_python/` |

---

## 二、依赖清单

### 必须依赖（pip install 即可）

```
numpy>=1.20        # 矩阵运算、FFT、向量化计算
scipy>=1.7         # 信号处理（lfilter, hilbert, correlate, convolve2d）
soundfile>=0.10    # WAV 音频读取（基于 libsndfile，支持多格式）
libsvm>=3.24       # SVR 模型加载与推理（Audio 模式）
```

### 不需要的（对比 C++ 版省去的依赖）

| 省去的 C++ 依赖 | 原因 |
|-----------------|------|
| Bazel 构建系统 | Python 无需编译 |
| Armadillo | numpy 替代 |
| PFFFT + SIMD | numpy.fft 替代 |
| TensorFlow / TFLite | 用多项式映射替代 |
| Abseil | Python 原生异常/logging 替代 |
| Protobuf | 用 dataclass 替代 |
| pybind11 | 原生 Python，无需绑定 |
| Google Test | 用 pytest 替代 |

---

## 三、项目目录结构

```
visqol_python/
├── IMPLEMENTATION_PLAN.md          # 本文件
├── setup.py                        # 包安装脚本
├── requirements.txt                # pip 依赖
│
├── visqol/                         # 主包
│   ├── __init__.py                 # 导出公共 API
│   ├── api.py                      # 公共 API 入口
│   ├── visqol_manager.py           # 流程编排器
│   ├── visqol_core.py              # 核心相似度算法
│   │
│   ├── # ---- 信号处理层 ----
│   ├── audio_utils.py              # 音频 I/O、SPL、dB 转换、单声道混合
│   ├── signal_utils.py             # 包络（Hilbert）、互相关、归一化
│   ├── analysis_window.py          # Hann 窗参数
│   │
│   ├── # ---- 频谱图层 ----
│   ├── gammatone.py                # ERB 系数 + Gammatone 滤波器组 + 频谱图构建
│   ├── spectrogram.py              # 频谱图数据结构（dB 转换、噪声底处理）
│   │
│   ├── # ---- Patch 层 ----
│   ├── patch_creator.py            # ImagePatchCreator + VadPatchCreator
│   ├── patch_selector.py           # 动态规划 Patch 匹配 + 精细对齐
│   │
│   ├── # ---- 相似度与质量映射层 ----
│   ├── nsim.py                     # NSIM（Neurogram Similarity Index Measure）
│   ├── quality_mapper.py           # SVR 映射 + 多项式/指数映射
│   │
│   └── # ---- 数据结构 ----
│       alignment.py                # 全局信号对齐
│
├── model/                          # 模型文件（从 C++ 版复制）
│   └── libsvm_nu_svr_model.txt     # 默认 SVR 模型
│
└── tests/                          # 测试
    ├── conftest.py                 # pytest fixtures（testdata 路径等）
    ├── test_conformance.py         # 一致性测试（核心验收标准）
    ├── test_gammatone.py           # Gammatone 频谱图单元测试
    ├── test_nsim.py                # NSIM 单元测试
    ├── test_alignment.py           # 对齐单元测试
    └── test_patch_selector.py      # DP Patch 匹配测试
```

---

## 四、分阶段实施计划

### 阶段 1：基础设施层（预计 0.5 天）

**目标**：项目骨架 + 音频 I/O + 基础数学工具

#### 1.1 项目初始化
- [ ] 创建目录结构
- [ ] `setup.py` + `requirements.txt`
- [ ] `visqol/__init__.py`

#### 1.2 `audio_utils.py` — 音频工具
对应 C++ 文件：`wav_reader.cc` + `misc_audio.cc` 部分

| 函数 | 功能 | 实现方案 |
|------|------|---------|
| `load_audio(path) -> (data, sr)` | 加载 WAV 文件，归一化到 [-1,1] | `soundfile.read()` |
| `to_mono(data) -> data` | 多通道混合为单声道 | `np.mean(data, axis=1)` |
| `load_as_mono(path) -> AudioSignal` | 加载 + 单声道 | 组合上述两者 |
| `calc_spl(signal) -> float` | 声压级 (dB SPL) | `20*log10(rms/2e-5)` |
| `scale_to_match_spl(ref, deg) -> deg_scaled` | 将 deg 缩放至与 ref 相同 SPL | `deg * 10^((ref_spl-deg_spl)/20)` |

**关键常量**：
```python
SPL_REFERENCE_POINT = 2e-5  # 20µPa
```

#### 1.3 `signal_utils.py` — 信号处理工具
对应 C++ 文件：`envelope.cc` + `xcorr.cc` + `misc_math.cc`

| 函数 | 功能 | 实现方案 |
|------|------|---------|
| `upper_envelope(signal) -> env` | 上包络（Hilbert 变换） | `scipy.signal.hilbert` |
| `find_best_lag(ref, deg) -> lag` | FFT 互相关找最佳延迟 | `scipy.signal.correlate` |
| `normalize(x) -> x_norm` | 归一化到 [0,1] | `x / np.max(np.abs(x))` |
| `next_pow2(n) -> int` | 下一个 2 的幂 | 位运算 |

#### 1.4 `analysis_window.py` — 分析窗
对应 C++ 文件：`analysis_window.cc/h`

```python
@dataclass
class AnalysisWindow:
    sample_rate: int
    overlap: float = 0.25
    duration: float = 0.08  # 秒

    @property
    def size(self) -> int:
        return round(self.sample_rate * self.duration)

    @property
    def hop_size(self) -> int:
        return int(self.size * self.overlap)

    @cached_property
    def window(self) -> np.ndarray:
        # Hann 窗: 0.5 - 0.5*cos(2πn/(N-1))
        return np.hanning(self.size)
```

**验收标准**：
- 能正确加载 testdata 中的 WAV 文件
- SPL 计算与 C++ 版一致（误差 < 0.001 dB）

---

### 阶段 2：Gammatone 频谱图（预计 2 天）

**目标**：精确实现 Gammatone 滤波器组频谱图，这是最关键也最复杂的模块

#### 2.1 `gammatone.py` — ERB + 滤波器 + 频谱图构建
对应 C++ 文件：`equivalent_rectangular_bandwidth.cc` + `gammatone_filterbank.cc` + `gammatone_spectrogram_builder.cc` + `signal_filter.cc`

##### 2.1.1 ERB 滤波器系数计算（`make_erb_filters`）

**硬编码常量**（Glasberg & Moore 参数）：
```python
EAR_Q = 9.26449
MIN_BW = 24.7
ORDER = 1.0
```

**中心频率计算**：
```python
def calc_center_freqs(num_channels, low_freq, high_freq):
    a = -(EAR_Q * MIN_BW)  # -228.833...
    b = -np.log(high_freq + EAR_Q * MIN_BW)
    c = np.log(low_freq + EAR_Q * MIN_BW)
    d = high_freq + EAR_Q * MIN_BW
    e = (b + c) / num_channels
    cfs = a + np.exp(np.arange(1, num_channels + 1) * e) * d
    return cfs
```

**带宽与滤波器系数**：
```python
erb = cfs / EAR_Q + MIN_BW  # (order=1 简化)
B = 1.019 * 2 * np.pi * erb
T = 1.0 / sample_rate

# 10 行系数矩阵: [A0, A11, A12, A13, A14, A2, B0, B1, B2, gain]
# 详细公式见 equivalent_rectangular_bandwidth.cc
```

**增益计算**（复数运算，最复杂的部分）：
```python
# 需精确复现 C++ 中的复数增益公式
# gain = abs(x1 * x2 * x3 * x4) / abs(x5 ^ 4)
# 其中 x1..x5 涉及 exp、cos、sin 的复数组合
```

##### 2.1.2 Gammatone 滤波器组（`GammatoneFilterBank`）

```python
class GammatoneFilterBank:
    def __init__(self, num_bands, min_freq):
        self.num_bands = num_bands
        self.min_freq = min_freq
        # 4 组滤波器状态 (每帧重置)
        self.filter_conditions = [None] * 4

    def apply_filter(self, signal, coeffs):
        """四阶级联 IIR 滤波"""
        # 等效于 4 次 scipy.signal.lfilter 级联
        # 每次使用不同的 A/B 系数对
        for stage in range(4):
            b = [coeffs.A0[band], coeffs.A1x[band], coeffs.A2[band]]
            a = [coeffs.B0[band], coeffs.B1[band], coeffs.B2[band]]
            signal, zi = scipy.signal.lfilter(b, a, signal, zi=self.filter_conditions[stage])
        return signal
```

##### 2.1.3 频谱图构建（`build_spectrogram`）

```python
def build_spectrogram(signal, sample_rate, window, num_bands, min_freq, speech_mode):
    max_freq = 8000.0 if speech_mode else sample_rate / 2.0
    coeffs = make_erb_filters(sample_rate, num_bands, min_freq, max_freq)
    coeffs = flip_updown(coeffs)  # 翻转行序

    hop_size = window.hop_size
    num_cols = 1 + (len(signal) - window.size) // hop_size
    spectrogram = np.zeros((num_bands, num_cols))

    for col in range(num_cols):
        frame = signal[col * hop_size : col * hop_size + window.size]
        frame = frame * window.window  # Hann 加窗

        # 重置滤波器状态（每帧都重置）
        filterbank.reset()

        # 逐频带应用 4 阶级联 Gammatone 滤波器
        filtered = filterbank.apply_filter(frame, coeffs)

        # 每频带 RMS = sqrt(mean(filtered^2))
        spectrogram[:, col] = np.sqrt(np.mean(filtered**2, axis=1))

    return spectrogram, center_freqs
```

**验收标准**：
- 对 `testdata/guitar48_stereo.wav` 构建的频谱图形状正确
- 中心频率与 C++ 版一致（误差 < 0.01 Hz）
- 频谱图 RMS 值与 C++ 版一致（相对误差 < 1%）

---

### 阶段 3：频谱图后处理 + Patch 创建（预计 1 天）

#### 3.1 `spectrogram.py` — 频谱图数据结构与处理
对应 C++ 文件：`spectrogram.cc` + `misc_audio.cc::PrepareSpectrogramsForComparison`

##### dB 转换
```python
def to_db(matrix):
    """10 * log10(|x|)，零值用 epsilon 替代"""
    abs_matrix = np.abs(matrix)
    abs_matrix[abs_matrix == 0] = np.finfo(float).eps
    return 10 * np.log10(abs_matrix)
```

##### 噪声底处理（`prepare_spectrograms_for_comparison`）
```python
NOISE_FLOOR_ABSOLUTE_DB = -45.0
NOISE_FLOOR_RELATIVE_TO_PEAK_DB = 45.0

def prepare_spectrograms_for_comparison(ref_spec, deg_spec):
    # 1. 转 dB
    ref_db = to_db(ref_spec)
    deg_db = to_db(deg_spec)

    # 2. 绝对噪声底: 将 < -45dB 的值抬到 -45
    ref_db = np.maximum(ref_db, NOISE_FLOOR_ABSOLUTE_DB)
    deg_db = np.maximum(deg_db, NOISE_FLOOR_ABSOLUTE_DB)

    # 3. 每帧相对噪声底: 找 ref/deg 合并的最大值, 减 45dB 作为地板
    for col in range(ref_db.shape[1]):
        any_max = max(ref_db[:, col].max(), deg_db[:, col].max())
        floor_db = any_max - NOISE_FLOOR_RELATIVE_TO_PEAK_DB
        ref_db[:, col] = np.maximum(ref_db[:, col], floor_db)
        deg_db[:, col] = np.maximum(deg_db[:, col], floor_db)

    # 4. 全局归一化: 减去全局最小值
    lowest = min(ref_db.min(), deg_db.min())
    ref_db -= lowest
    deg_db -= lowest

    return ref_db, deg_db
```

#### 3.2 `patch_creator.py` — Patch 创建
对应 C++ 文件：`image_patch_creator.cc` + `vad_patch_creator.cc` + `rms_vad.h`

##### Audio 模式 Patch（等间距）
```python
def create_ref_patch_indices_audio(num_spectrogram_cols, patch_size):
    """从半个 patch 处开始，每隔一个 patch 取一个索引"""
    start = patch_size // 2
    indices = list(range(start, num_spectrogram_cols, patch_size))
    return indices

def create_patches(spectrogram, indices, patch_size):
    """从频谱图中按索引切出 patches"""
    patches = []
    half = patch_size // 2
    for idx in indices:
        patch = spectrogram[:, idx - half : idx - half + patch_size]
        patches.append(patch)
    return patches
```

##### Speech 模式 Patch（VAD 过滤）

RMS VAD 实现：
```python
def rms_vad(signal, sample_rate, frame_size):
    """基于 RMS 的简易 VAD"""
    # 1. 量化到 int16 范围
    quantized = np.round(signal * 32768).astype(np.int16)
    # 2. 分帧计算 RMS
    # 3. 超过阈值的帧标记为活动
    # 阈值逻辑: RMS 中位数 * 因子
    ...
```

VadPatchCreator：
```python
def create_ref_patch_indices_speech(signal, sample_rate, spectrogram_cols, patch_size):
    """先做 VAD，只保留有语音活动的 patch"""
    activity = rms_vad(signal, sample_rate, ...)
    # 归一化信号
    # 计算均匀间距 patch 索引
    # 过滤: 只保留至少 1 帧有活动的 patch
    return filtered_indices
```

**验收标准**：
- Audio 模式 patch 数量与 C++ 版一致
- Speech 模式 VAD 过滤后 patch 数量与 C++ 版一致（允许 ±1 个 patch）

---

### 阶段 4：对齐 + NSIM 相似度（预计 1.5 天）

#### 4.1 `alignment.py` — 全局对齐
对应 C++ 文件：`alignment.cc`

```python
def globally_align(ref_signal, deg_signal):
    """全局对齐退化信号到参考信号"""
    # 1. 计算上包络
    ref_env = upper_envelope(ref_signal)
    deg_env = upper_envelope(deg_signal)

    # 2. 互相关找最佳延迟
    best_lag = find_best_lag(ref_env, deg_env)

    # 3. 对齐: 正 lag 表示 deg 需要截断前面
    if best_lag > 0:
        aligned_deg = np.pad(deg_signal[best_lag:], (0, best_lag))
    else:
        aligned_deg = np.pad(deg_signal, (-best_lag, 0))[:len(deg_signal)]

    return aligned_deg, best_lag

def align_and_truncate(ref, deg):
    """对齐后截断到相同长度"""
    aligned_deg, lag = globally_align(ref, deg)
    min_len = min(len(ref), len(aligned_deg))
    return ref[:min_len], aligned_deg[:min_len], lag
```

#### 4.2 `nsim.py` — NSIM 相似度度量
对应 C++ 文件：`neurogram_similiarity_index_measure.cc` + `convolution_2d.cc`

**这是最关键的质量度量算法**

```python
# 3x3 高斯窗权重（硬编码）
GAUSSIAN_WINDOW = np.array([
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
    [0.0838251475442633, 0.619485845753726,  0.0838251475442633],
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052]
])

# 常量
C1 = (0.01 * 1.0) ** 2   # = 0.0001
C3 = (0.03 * 1.0) ** 2 / 2  # = 0.00045

def conv2d_with_boundary(kernel, matrix):
    """带边界复制填充的 valid 卷积"""
    padded = np.pad(matrix, 1, mode='edge')  # 复制边界
    return scipy.signal.convolve2d(padded, kernel, mode='valid')

def measure_patch_similarity(ref_patch, deg_patch):
    """计算一对 patch 的 NSIM"""
    w = GAUSSIAN_WINDOW

    # 局部均值
    mu_r = conv2d_with_boundary(w, ref_patch)
    mu_d = conv2d_with_boundary(w, deg_patch)

    # 亮度分量
    intensity = (2 * mu_r * mu_d + C1) / (mu_r**2 + mu_d**2 + C1)

    # 方差与协方差
    sigma_r_sq = conv2d_with_boundary(w, ref_patch**2) - mu_r**2
    sigma_d_sq = conv2d_with_boundary(w, deg_patch**2) - mu_d**2
    sigma_rd = conv2d_with_boundary(w, ref_patch * deg_patch) - mu_r * mu_d

    # 结构分量（处理负方差的特殊情况）
    var_product = sigma_r_sq * sigma_d_sq
    denom = np.where(var_product < 0, C3, np.sqrt(var_product) + C3)
    structure = (sigma_rd + C3) / denom

    # 合成
    sim_map = intensity * structure

    # 输出
    freq_band_means = np.mean(sim_map, axis=1)     # 每频带均值
    freq_band_stddevs = np.std(sim_map, axis=1)     # 每频带标准差
    deg_energy = np.mean(deg_patch, axis=1)          # 退化信号每频带能量
    similarity = np.mean(freq_band_means)            # 总相似度

    return PatchSimilarityResult(
        similarity=similarity,
        freq_band_means=freq_band_means,
        freq_band_stddevs=freq_band_stddevs,
        freq_band_deg_energy=deg_energy,
    )
```

**验收标准**：
- 给定相同的 ref/deg patch 矩阵，NSIM 值与 C++ 版一致（误差 < 0.001）
- 边界处理（边缘复制填充）行为与 C++ 版一致

---

### 阶段 5：动态规划 Patch 匹配（预计 1.5 天）

#### 5.1 `patch_selector.py` — DP 匹配 + 精细对齐
对应 C++ 文件：`comparison_patches_selector.cc`（384 行，最复杂的模块之一）

##### DP 前向传播
```python
DEFAULT_SEARCH_WINDOW_RADIUS = 60  # 默认搜索半径

def find_most_optimal_deg_patches(ref_patches, ref_indices, deg_spectrogram,
                                   patch_size, search_window_radius=60):
    """动态规划搜索最优退化 patch 匹配"""
    num_patches = len(ref_patches)
    num_frames = patch_size
    search_window = search_window_radius * num_frames
    spectrogram_cols = deg_spectrogram.shape[1]

    # DP 表
    cumulative_dp = defaultdict(lambda: -np.inf)  # [patch_idx][slide_offset]
    backtrace = {}  # [patch_idx][slide_offset] -> past_offset

    for patch_idx in range(num_patches):
        ref_frame = ref_indices[patch_idx]
        low = max(0, ref_frame - search_window)
        high = min(spectrogram_cols, ref_frame + search_window)

        for slide_offset in range(low, high):
            deg_patch = build_degraded_patch(deg_spectrogram, slide_offset, patch_size)
            sim = measure_patch_similarity(ref_patches[patch_idx], deg_patch)

            if patch_idx > 0:
                # 向前搜索: 找前一个 patch 最优累积分
                prev_low = max(0, ref_indices[patch_idx-1] - search_window)
                best_prev_sim = -np.inf
                best_prev_offset = -1
                for back in range(slide_offset - 1, prev_low - 1, -1):
                    if cumulative_dp[(patch_idx-1, back)] > best_prev_sim:
                        best_prev_sim = cumulative_dp[(patch_idx-1, back)]
                        best_prev_offset = back

                sim.similarity += best_prev_sim

                # 丢包处理
                if cumulative_dp[(patch_idx-1, slide_offset)] > sim.similarity:
                    sim.similarity = cumulative_dp[(patch_idx-1, slide_offset)]
                    best_prev_offset = slide_offset

            cumulative_dp[(patch_idx, slide_offset)] = sim.similarity
            backtrace[(patch_idx, slide_offset)] = best_prev_offset

    # 回溯
    ...
```

> **优化说明**：上述嵌套循环在 Python 中较慢，但因 patch 数量通常 < 50、搜索窗口 ~1800，实际计算量可控（约百万次 NSIM 计算）。如果性能不可接受，可用 `@numba.jit` 加速 DP 内层循环。

##### 精细对齐
```python
def finely_align_and_recreate_patches(results, ref_signal, deg_signal,
                                       spectrogram_builder, window):
    """对每对匹配的 patch 做时域精细对齐"""
    for i, result in enumerate(results):
        if result.similarity == 0:
            continue  # 跳过丢包 patch

        # 1. 按时间范围从原始音频中切片
        ref_slice = slice_signal(ref_signal, result.ref_start, result.ref_end)
        deg_slice = slice_signal(deg_signal, result.deg_start, result.deg_end)

        # 2. 精细对齐
        aligned_ref, aligned_deg, lag = align_and_truncate(ref_slice, deg_slice)

        # 3. 重建频谱图
        ref_spec = spectrogram_builder.build(aligned_ref, ...)
        deg_spec = spectrogram_builder.build(aligned_deg, ...)
        ref_spec, deg_spec = prepare_spectrograms_for_comparison(ref_spec, deg_spec)

        # 4. 重新计算 NSIM
        new_sim = measure_patch_similarity(ref_spec, deg_spec)

        # 5. 取最大值: 只在精细对齐改善结果时才更新
        if new_sim.similarity > result.similarity:
            results[i] = new_sim
            # 更新时间戳（根据 lag 调整）

    return results
```

**验收标准**：
- DP 回溯路径与 C++ 版一致
- 精细对齐的 lag 值与 C++ 版一致

---

### 阶段 6：质量映射器（预计 0.5 天）

#### 6.1 `quality_mapper.py` — SVR + 多项式映射
对应 C++ 文件：`svr_similarity_to_quality_mapper.cc` + `speech_similarity_to_quality_mapper.cc` + `support_vector_regression_model.cc`

##### SVR 映射器（Audio 模式）
```python
from svmutil import svm_load_model, svm_predict

class SvrQualityMapper:
    def __init__(self, model_path):
        self.model = svm_load_model(model_path)

    def predict(self, fvnsim, fvnsim10=None, fstdnsim=None, fvdegenergy=None):
        """SVR 只使用 fvnsim 向量作为输入"""
        # libsvm 格式: {1: val1, 2: val2, ...}
        x = {i+1: v for i, v in enumerate(fvnsim)}
        _, pred, _ = svm_predict([0], [x], self.model, '-q')
        return np.clip(pred[0], 1.0, 5.0)
```

##### 多项式/指数映射器（Speech 模式）
```python
# 硬编码拟合参数
FIT_A = -262.847869
FIT_B = 0.0154302525
FIT_X0 = -361.063949
FIT_SCALE = 1.245063

class SpeechQualityMapper:
    def __init__(self, scale_to_max_mos=True):
        self.scale = FIT_SCALE if scale_to_max_mos else 1.0

    def predict(self, fvnsim, fvnsim10=None, fstdnsim=None, fvdegenergy=None):
        nsim_mean = np.mean(fvnsim)
        mos = FIT_A + np.exp(FIT_B * (nsim_mean - FIT_X0))
        return np.clip(mos * self.scale, 1.0, 5.0)
```

**验收标准**：
- SVR 加载 `model/libsvm_nu_svr_model.txt` 后，给定相同 fvnsim 向量，输出 MOS 一致
- 多项式映射公式验算正确

---

### 阶段 7：核心编排 + 公共 API（预计 1 天）

#### 7.1 `visqol_core.py` — 核心算法
对应 C++ 文件：`visqol.cc`

```python
class VisqolCore:
    def calculate_similarity(self, ref_signal, deg_signal, ...):
        """完整的相似度计算流水线"""
        # 1. SPL 匹配
        deg_scaled = scale_to_match_spl(ref_signal, deg_signal)

        # 2. 构建频谱图
        ref_spec = self.spec_builder.build(ref_signal, ...)
        deg_spec = self.spec_builder.build(deg_scaled, ...)

        # 3. 频谱图预处理
        ref_spec, deg_spec = prepare_spectrograms_for_comparison(ref_spec, deg_spec)

        # 4. 创建参考 patches
        ref_indices = self.patch_creator.create_indices(ref_spec, ...)
        ref_patches = create_patches(ref_spec, ref_indices, ...)

        # 5. DP 匹配
        results = find_most_optimal_deg_patches(ref_patches, ref_indices, deg_spec, ...)

        # 6. 精细对齐（可选）
        if not self.disable_realignment:
            results = finely_align_and_recreate_patches(results, ...)

        # 7. 聚合统计量
        fvnsim = calc_mean_freq_band_means(results)
        fvnsim10 = calc_freq_band_quantile(results, quantile=0.10)
        fstdnsim = calc_mean_freq_band_stddevs(results, ...)
        fvdegenergy = calc_mean_freq_band_deg_energy(results)

        # 8. 预测 MOS
        moslqo = self.quality_mapper.predict(fvnsim, fvnsim10, fstdnsim, fvdegenergy)

        # 9. 极端值处理
        vnsim = np.mean(fvnsim)
        if vnsim < 0.15:
            moslqo = 1.0

        return SimilarityResult(moslqo=moslqo, vnsim=vnsim, fvnsim=fvnsim, ...)
```

**聚合统计量的精确公式**：

```python
def calc_mean_freq_band_means(results):
    """fvnsim: 每频带跨 patch 的均值"""
    all_means = np.array([r.freq_band_means for r in results])
    return np.mean(all_means, axis=0)

def calc_freq_band_quantile(results, quantile=0.10):
    """fvnsim10: 排序后取底部 10% 的均值"""
    all_means = np.array([r.freq_band_means for r in results])
    num_in_quantile = max(1, int(np.floor(len(results) * quantile)))
    result = np.zeros(all_means.shape[1])
    for band in range(all_means.shape[1]):
        sorted_vals = np.sort(all_means[:, band])
        result[band] = np.mean(sorted_vals[:num_in_quantile])
    return result

def calc_mean_freq_band_stddevs(results, frame_duration, fvnsim):
    """fstdnsim: 池化方差（pooled variance）"""
    # 标准的池化方差公式
    total_frames = 0
    contributions = np.zeros(num_bands)
    for r in results:
        n = int(np.ceil(r.patch_duration / frame_duration))
        total_frames += n
        contributions += (n - 1) * r.freq_band_stddevs**2 + n * r.freq_band_means**2
    variance = (contributions - fvnsim**2 * total_frames) / (total_frames - 1)
    return np.where(variance < 0, 0.0, np.sqrt(variance))
```

#### 7.2 `visqol_manager.py` — 流程编排器
对应 C++ 文件：`visqol_manager.cc`

```python
class VisqolManager:
    # 默认参数
    PATCH_SIZE_AUDIO = 30
    PATCH_SIZE_SPEECH = 20
    NUM_BANDS_AUDIO = 32
    NUM_BANDS_SPEECH = 21
    MIN_FREQ = 50.0
    OVERLAP = 0.25
    DURATION_MISMATCH_TOLERANCE = 1.0  # 秒

    def __init__(self, use_speech_mode=False, search_window=60, ...):
        ...

    def run(self, ref_path, deg_path):
        """完整的运行流程"""
        # 1. 加载音频
        ref = load_as_mono(ref_path)
        deg = load_as_mono(deg_path)

        # 2. 验证
        assert ref.sample_rate == deg.sample_rate
        assert abs(ref.duration - deg.duration) < self.DURATION_MISMATCH_TOLERANCE

        # 3. 全局对齐
        ref_aligned, deg_aligned, _ = align_and_truncate(ref.data, deg.data)

        # 4. 核心计算
        return self.visqol_core.calculate_similarity(ref_aligned, deg_aligned, ...)
```

#### 7.3 `api.py` — 公共 API
对应 C++ 文件：`visqol_api.cc`

```python
class VisqolApi:
    """ViSQOL 的公共 Python API"""

    def create(self, mode="audio", model_path=None, **kwargs):
        """初始化 ViSQOL"""
        ...

    def measure(self, ref_path, deg_path):
        """比较两个音频文件，返回 MOS-LQO 分数"""
        return self.manager.run(ref_path, deg_path)

    def measure_from_arrays(self, ref_array, deg_array, sample_rate):
        """直接从 numpy 数组比较"""
        ...
```

**验收标准**：整个 API 能运行，输出 MOS 分数

---

### 阶段 8：一致性测试与调优（预计 2 天）

#### 8.1 一致性测试
对应 C++ 文件：`tests/conformance_test.cc` + `src/include/conformance.h`

##### 测试用例（Speech 模式，使用多项式映射）

| # | Reference | Degraded | 预期 MOS (exponential) | 容差 |
|---|-----------|----------|----------------------|------|
| 1 | `CA01_01.wav` | `transcoded_CA01_01.wav` | 3.374505555111911 | ±0.05 |
| 2 | `CA01_01.wav` | `CA01_01.wav` (self) | — (需标定) | ±0.05 |

> 注意：由于我们使用多项式映射（非 TFLite lattice），Speech 模式的预期分数应参考 `kConformance*Exponential` 系列常量。

##### 测试用例（Audio 模式，SVR 映射）

| # | Reference | Degraded | 预期 MOS | 容差 |
|---|-----------|----------|---------|------|
| 1 | `strauss48_stereo.wav` | `strauss48_stereo_lp35.wav` | 1.389 | ±0.05 |
| 2 | `steely48_stereo.wav` | `steely48_stereo_lp7.wav` | 2.250 | ±0.05 |
| 3 | `sopr48_stereo.wav` | `sopr48_stereo_256kbps_aac.wav` | 4.682 | ±0.05 |
| 4 | `ravel48_stereo.wav` | `ravel48_stereo_128kbps_opus.wav` | 4.465 | ±0.05 |
| 5 | `moonlight48_stereo.wav` | `moonlight48_stereo_128kbps_aac.wav` | 4.684 | ±0.05 |
| 6 | `harpsichord48_stereo.wav` | `harpsichord48_stereo_96kbps_mp3.wav` | 4.224 | ±0.05 |
| 7 | `guitar48_stereo.wav` | `guitar48_stereo_64kbps_aac.wav` | 4.350 | ±0.05 |
| 8 | `glock48_stereo.wav` | `glock48_stereo_48kbps_aac.wav` | 4.332 | ±0.05 |
| 9 | `contrabassoon48_stereo.wav` | `contrabassoon48_stereo_24kbps_aac.wav` | 2.347 | ±0.05 |
| 10 | `castanets48_stereo.wav` | `castanets48_stereo.wav` (identity) | 4.732 | ±0.05 |

#### 8.2 调试策略

如果一致性测试不通过，按以下顺序排查：

1. **频谱图对比**：对比 C++ 和 Python 的频谱图矩阵（逐元素差异）
2. **Patch 对比**：对比 patch 索引、patch 数量
3. **NSIM 对比**：对比单个 patch 对的 NSIM 值
4. **DP 路径对比**：对比 DP 回溯选择的 offset
5. **统计量对比**：对比 fvnsim、fvnsim10 向量
6. **最终 MOS 对比**：SVR/多项式映射结果

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Gammatone 滤波器数值精度 | 频谱图差异 → MOS 偏移 | 逐频带对比中间结果 |
| IIR 滤波器状态重置时机 | 频谱图全局偏差 | 确认每帧重置（C++ 代码确认是如此） |
| libsvm Python 绑定兼容性 | SVR 预测结果不一致 | 使用同一 libsvm 版本 (3.24) |
| DP 搜索性能 | Python 循环太慢 | 向量化 NSIM 计算；必要时 numba 加速 |
| `scipy.signal.lfilter` 精度 | 滤波结果微小差异 | 使用 `float64`，与 C++ 的 `double` 一致 |
| 边界条件差异 | Patch 构建/卷积边缘效应 | 精确复现 C++ 的零填充和 edge-pad 逻辑 |

---

## 六、性能预估

| 操作 | C++ 耗时 | Python 预估 | 说明 |
|------|---------|------------|------|
| WAV 加载 (10s, 48kHz) | ~10ms | ~20ms | soundfile 基于 libsndfile |
| Gammatone 频谱图 | ~100ms | ~500ms | scipy.lfilter 逐频带循环 |
| DP Patch 匹配 | ~200ms | ~2-5s | 嵌套循环，可 numba 加速 |
| NSIM 计算 (单 patch) | ~0.1ms | ~0.5ms | scipy.convolve2d |
| 精细对齐 | ~100ms | ~1s | 逐 patch 重建频谱图 |
| **总计 (10s 音频)** | **~0.5s** | **~5-10s** | 可接受范围 |

---

## 七、后续扩展（可选）

1. **TFLite 支持**：如果后续需要更高精度的 Speech 模式，可加入 `tflite-runtime` 依赖
2. **Numba 加速**：对 DP 搜索和 Gammatone 滤波的热循环使用 `@numba.jit`
3. **命令行工具**：添加 `__main__.py` 支持 `python -m visqol` 调用
4. **流式处理**：支持大文件分段处理
5. **并行计算**：多线程/多进程处理批量音频对

---

## 八、交付物检查清单

- [ ] 所有 Python 源代码文件
- [ ] `requirements.txt` + `setup.py`
- [ ] 模型文件复制 (`model/libsvm_nu_svr_model.txt`)
- [ ] 一致性测试脚本 + 通过所有测试（MOS 差异 < 0.05）
- [ ] 使用示例文档

---

*计划版本: v1.0*
*最后更新: 2026-03-23*
