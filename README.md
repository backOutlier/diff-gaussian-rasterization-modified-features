# diff-gaussian-rasterization-modified-features

A modified version of [dcharatan/diff-gaussian-rasterization-modified](https://github.com/dcharatan/diff-gaussian-rasterization-modified) that adds **per-Gaussian language feature rendering** while preserving the **e3nn-compatible spherical harmonics ordering**.

---

## Modifications

This repo builds on top of `dcharatan/diff-gaussian-rasterization-modified` and ports the language feature rendering capability from [`ngailapdi/diff-gaussian-rasterization-w-depth-feature`](https://github.com/ngailapdi/diff-gaussian-rasterization-w-depth-feature).

| Repo | SH ordering | Feature rendering |
|------|------------|-------------------|
| dcharatan/diff-gaussian-rasterization-modified | e3nn (x,y,z) ✅ | ❌ |
| ngailapdi/diff-gaussian-rasterization-w-depth-feature | original (y,z,x) ❌ | ✅ |
| **This repo** | **e3nn (x,y,z) ✅** | **✅** |

Key changes:
- **Spherical harmonics** use the e3nn convention (x, y, z ordering), compatible with [dcharatan's version](https://github.com/dcharatan/diff-gaussian-rasterization-modified).
- **Language feature alpha-blending**: each Gaussian carries a `NUM_CHANNELS_language_feature`-dimensional feature vector (default **128**, matching a dual-branch AE bottleneck). Features are alpha-composited per pixel alongside RGB.
- **Backward pass** correctly propagates gradients through the feature channel.
- **Python API** updated with an optional `include_feature` flag (defaults to `False` for drop-in compatibility).

For a fully latent feature rendering variant, also check out [Chrixtar/latent-gaussian-rasterization](https://github.com/Chrixtar/latent-gaussian-rasterization).

---

## Customizing Feature Dimension

The feature channel count is set in `cuda_rasterizer/config.h`:

```c
#define NUM_CHANNELS_language_feature 128  // change to match your model
```

Rebuild the extension after any change to this value.

---

## Installation

```bash
pip install -e .
```

Requires a CUDA-capable GPU and compatible PyTorch installation.

---

## Usage

```python
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

# Standard RGB rendering (feature rendering disabled by default)
rendered, radii = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=None,
    opacities=opacities,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
)

# With language feature rendering
rendered, radii, rendered_feature = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=None,
    opacities=opacities,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
    language_feature_precomp=language_features,  # (N, 128)
    include_feature=True,
)
```

---

## Original Paper

This rasterizer is based on the engine from "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". Please cite the original work if you use this in your research:

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
