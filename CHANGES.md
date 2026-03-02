# diff-gaussian-rasterization-modified 改动说明

> 基于 `dcharatan/diff-gaussian-rasterization-modified`，移植 `ngailapdi/diff-gaussian-rasterization-w-depth-feature` 的语言特征渲染能力（当前配置为 **128 维**，匹配 AE bottleneck），同时保留 dcharatan 的 **e3nn SH 球谐系数排列方式**（x,y,z 顺序）。

---

## 背景

| 仓库 | SH 排列 | 特征渲染 |
|------|---------|---------|
| dcharatan/diff-gaussian-rasterization-modified | e3nn (x,y,z) ✅ | 无 ❌ |
| ngailapdi/diff-gaussian-rasterization-w-depth-feature | 原版 (y,z,x) ❌ | 有 ✅ |

两者 SH 排列不兼容，不能直接替换。本次改动选择在 dcharatan 版本的基础上，**仅移植特征渲染部分**。

---

## 修改文件总览

| 文件 | 改动类型 |
|------|---------|
| `cuda_rasterizer/config.h` | 新增特征维度宏定义 |
| `cuda_rasterizer/forward.h` | 前向声明增加特征参数 |
| `cuda_rasterizer/forward.cu` | 前向核函数增加特征 alpha 混合 |
| `cuda_rasterizer/backward.h` | 反向声明增加特征梯度参数 |
| `cuda_rasterizer/backward.cu` | 反向核函数增加特征梯度传播 |
| `cuda_rasterizer/rasterizer.h` | 公共接口增加特征参数（带默认值） |
| `cuda_rasterizer/rasterizer_impl.cu` | 实现层透传新参数 |
| `rasterize_points.h` | Python 桥接层声明更新 |
| `rasterize_points.cu` | Python 桥接层分配特征缓冲区 |
| `diff_gaussian_rasterization/__init__.py` | Python API 更新 |

---

## 逐文件改动详情

### 1. `cuda_rasterizer/config.h`

新增一行宏定义，指定语言特征的通道数：

```c
#define NUM_CHANNELS 3
#define NUM_CHANNELS_language_feature 128   // ← 新增（匹配 AE bottleneck 维度）
```

> **维度说明**：初始移植时为 256，后改为 128 以匹配 DualBranchSemanticAutoencoder 的 bottleneck 维度。修改方式见下方 [自定义特征维度](#自定义特征维度) 章节。

---

### 2. `cuda_rasterizer/forward.h`

`FORWARD::render` 函数声明末尾新增 3 个参数：

```cpp
// 新增参数
const float* language_feature,      // 每个 Gaussian 的特征向量 (N × F)，F = NUM_CHANNELS_language_feature
float* out_language_feature,         // 输出特征图 (F × H × W)
bool include_feature                 // 是否启用特征渲染
```

---

### 3. `cuda_rasterizer/forward.cu`

**核函数 `renderCUDA`** 内增加特征 alpha 混合逻辑：

```cpp
// 在 T（透射率）累积循环内
if (include_feature) {
    float F_acc[NUM_CHANNELS_language_feature] = {0};
    for (int ch = 0; ch < NUM_CHANNELS_language_feature; ch++) {
        // 访问全局内存，避免 shared memory 溢出
        F_acc[ch] += language_feature[global_id * 256 + ch] * alpha * T;
    }
    // 写入输出特征图
    if (inside) {
        for (int ch = 0; ch < NUM_CHANNELS_language_feature; ch++)
            out_language_feature[ch * H * W + pix_id] += F_acc[ch];
    }
}
```

> **设计说明**：特征直接从全局内存读取（而非 shared memory），避免 256 × 16 × 4 = 16KB/block 导致的 shared memory 溢出。

---

### 4. `cuda_rasterizer/backward.h`

`BACKWARD::render` 声明末尾新增 4 个参数：

```cpp
const float* language_feature,       // 前向时的特征输入 (N × F)
const float* dL_dpixels_feature,     // 来自上游的特征损失梯度 (F × H × W)
float* dL_dlanguage_feature,          // 输出：对每个 Gaussian 特征的梯度 (N × F)
bool include_feature
```

---

### 5. `cuda_rasterizer/backward.cu`

**反向核函数** 内增加特征梯度传播：

```cpp
if (include_feature) {
    // 遍历特征维度，计算 dL/d(language_feature[g])
    for (int ch = 0; ch < NUM_CHANNELS_language_feature; ch++) {
        float dchannel_dcolor = alpha * last_weight_f;
        float dL_dchannel_f = dL_dpixel_f[ch];
        atomicAdd(&dL_dlanguage_feature[global_id * 256 + ch],
                  dchannel_dcolor * dL_dchannel_f);
    }
    // 特征项也贡献到 dL/d(alpha)
    dL_dalpha += /* feature term */;
}
```

> **注意**：特征没有背景色项（RGB 有 `bg_color`），梯度计算略有不同。

---

### 6. `cuda_rasterizer/rasterizer.h`

公共接口新增参数，**全部带默认值**，保证向后兼容：

```cpp
// Rasterizer::forward 新增（末尾，带默认值）
const float* language_feature_precomp = nullptr,
float* out_language_feature = nullptr,
bool include_feature = false

// Rasterizer::backward 新增（末尾，带默认值）
const float* language_feature_precomp = nullptr,
const float* dL_dpix_feature = nullptr,
float* dL_dlanguage_feature = nullptr,
bool include_feature = false
```

---

### 7. `cuda_rasterizer/rasterizer_impl.cu`

更新 `Rasterizer::forward` 和 `Rasterizer::backward` 的函数签名，并将新参数透传给 `FORWARD::render` 和 `BACKWARD::render`。无算法逻辑改动。

---

### 8. `rasterize_points.h`

Python ↔ CUDA 桥接层声明更新：

- `RasterizeGaussiansCUDA` 返回值：6 元组 → **7 元组**（新增 `language_feature`）  
- `RasterizeGaussiansCUDA` 输入：新增 `language_feature`、`include_feature`  
- `RasterizeGaussiansBackwardCUDA` 返回值：8 元组 → **9 元组**（新增 `dL_dlanguage_feature`）  
- `RasterizeGaussiansBackwardCUDA` 输入：新增对应梯度参数

> **注意**：缓冲区大小由编译时常量 `NUM_CHANNELS_language_feature` 决定（当前 128），不能在运行时动态改变。

---

### 9. `rasterize_points.cu`

**前向桥接函数**：
```cpp
// 分配输出特征图缓冲区
torch::Tensor out_language_feature;
if (include_feature)
    out_language_feature = torch::full({256, H, W}, 0.0, options);
else
    out_language_feature = torch::empty({0}, options);

// 返回 7 元组
return std::make_tuple(rendered, out_color, out_language_feature,
                       radii, geomBuffer, binningBuffer, imgBuffer);
```

**反向桥接函数**：
```cpp
// 分配特征梯度缓冲区
torch::Tensor dL_dlanguage_feature;
if (include_feature)
    dL_dlanguage_feature = torch::zeros({P, 256}, means3D.options());
else
    dL_dlanguage_feature = torch::empty({0}, means3D.options());

// 返回 9 元组（在原 8 元组基础上追加）
```

---

### 10. `diff_gaussian_rasterization/__init__.py`

**`GaussianRasterizationSettings` dataclass** 新增字段：
```python
include_feature: bool = False   # 默认关闭，向后兼容
```

**`rasterize_gaussians()`** 新增参数：
```python
language_feature_precomp: Optional[Tensor] = None
```

**`_RasterizeGaussians.forward`** 返回值变更：
```python
# 原来
return color, radii

# 现在
return color, out_language_feature, radii
```

**`_RasterizeGaussians.backward`**：
```python
# 接收新增的梯度
def backward(ctx, grad_out_color, grad_out_language_feature, _):
    ...
    # 将 dL_dlanguage_feature 返回给 language_feature_precomp 的梯度位置
```

**`GaussianRasterizer.forward`** 新增参数：
```python
def forward(self, ..., language_feature_precomp=None):
```

---

## 使用方式

### 启用特征渲染（新功能）

```python
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

settings = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=sh_degree,
    campos=campos,
    prefiltered=False,
    debug=False,
    include_feature=True,   # ← 开启特征渲染
)

rasterizer = GaussianRasterizer(raster_settings=settings)

color, language_feature, radii = rasterizer(
    means3D=means3D,                         # (N, 3)
    means2D=means2D,                         # (N, 3)
    shs=shs,                                 # (N, K, 3) 或 None
    colors_precomp=colors_precomp,           # (N, 3) 或 None
    opacities=opacities,                     # (N, 1)
    scales=scales,                           # (N, 3)
    rotations=rotations,                     # (N, 4)
    cov3D_precomp=cov3D_precomp,
    language_feature_precomp=features,       # (N, 128) ← 每个 Gaussian 的特征
)

# color:            torch.Tensor (3, H, W)
# language_feature: torch.Tensor (128, H, W)
# radii:            torch.Tensor (N,)
```

### 仅 RGB 渲染（向后兼容）

```python
settings = GaussianRasterizationSettings(
    ...,
    include_feature=False,   # 默认值，可省略
)
rasterizer = GaussianRasterizer(raster_settings=settings)

# 需要用 3 元组解包（第二项为空 tensor）
color, _, radii = rasterizer(...)
```

---

## 关联修复：`depthsplat/src/model/decoder/cuda_splatting.py`

由于 rasterizer 返回值从 **2 元组** 变为 **3 元组**，原有 RGB 推理代码会抛出：
```
ValueError: too many values to unpack (expected 2)
```

已修复两处（L116、L209）：
```python
# 修复前
image, radii = rasterizer(...)

# 修复后
image, _, radii = rasterizer(...)
```

---

## 编译方式（RTX 5090 / sm_120）

```bash
cd /home/zhihaogu/ECCV/VLA-Adapter/others/diff-gaussian-rasterization-modified
git submodule update --init --recursive   # 确保 glm 子模块存在

conda activate vla-adapter
TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation
```

---

## 测试验证

运行测试脚本（4 项全部通过）：

```bash
python test_feature_rendering.py
```

| 测试项 | 结果 |
|--------|------|
| 仅 RGB 渲染（向后兼容） | ✅ PASSED |
| 特征渲染输出形状 `(128, 64, 64)` | ✅ PASSED |
| 反向传播梯度非零 | ✅ PASSED |
| 开启/关闭特征时 RGB 完全一致 | ✅ PASSED |

---

## 自定义特征维度

特征通道数由 `cuda_rasterizer/config.h` 中的编译时常量控制：

```c
#define NUM_CHANNELS_language_feature 128
```

所有 CUDA 核函数（forward/backward）、缓冲区分配（`rasterize_points.cu`）均通过此宏引用，**不存在硬编码的数值**。

### 修改步骤

如需更改特征维度（例如改为 64 或 256），只需：

**1. 修改 `cuda_rasterizer/config.h`（唯一需要改的文件）：**

```c
// 改成你需要的维度
#define NUM_CHANNELS_language_feature 64   // 或 256, 512 等
```

**2. 重新编译：**

```bash
cd /path/to/diff-gaussian-rasterization-modified
TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation
```

**3. Python 侧对应调整：**

```python
# language_feature_precomp 的第二维必须与编译时的宏一致
language_feature_precomp=features,  # (N, 64) 或 (N, 256) 等
```

### 维度变更历史

| 时间 | 值 | 原因 |
|------|-----|------|
| 初始移植 | 256 | 与 ngailapdi 原版一致 |
| 当前 | **128** | 匹配 DualBranchSemanticAutoencoder bottleneck 维度 |

> ⚠️ **注意**：传入 `language_feature_precomp` 的 tensor 第二维 **必须** 与编译时的 `NUM_CHANNELS_language_feature` 完全一致，否则 CUDA 核函数会越界访问内存导致 crash 或产生脏数据。
