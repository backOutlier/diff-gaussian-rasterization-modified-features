"""Test script to verify language feature rendering through Gaussian splatting."""
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def test_feature_rendering():
    device = torch.device("cuda")
    
    # Setup: 100 random Gaussians
    N = 100
    H, W = 64, 64
    
    means3D = torch.randn(N, 3, device=device) * 0.5
    means3D[:, 2] += 3.0  # push away from camera
    means2D = torch.zeros(N, 3, device=device)
    
    # SH coefficients (degree 0 only = 1 coeff per color channel)
    shs = torch.randn(N, 1, 3, device=device) * 0.5
    
    opacities = torch.sigmoid(torch.randn(N, 1, device=device))
    scales = torch.abs(torch.randn(N, 3, device=device)) * 0.1
    rotations = torch.randn(N, 4, device=device)
    rotations = rotations / rotations.norm(dim=1, keepdim=True)
    
    # Language features: (N, 256)
    language_features = torch.randn(N, 256, device=device, requires_grad=True)
    
    # Camera setup (identity-like)
    fovx = math.pi / 2
    fovy = math.pi / 2
    viewmatrix = torch.eye(4, device=device)
    projmatrix = torch.eye(4, device=device)
    # Simple perspective
    projmatrix[0, 0] = 1.0 / math.tan(fovx / 2)
    projmatrix[1, 1] = 1.0 / math.tan(fovy / 2)
    projmatrix[2, 2] = 100.0 / (100.0 - 0.1)
    projmatrix[2, 3] = 1.0
    projmatrix[3, 2] = -(100.0 * 0.1) / (100.0 - 0.1)
    projmatrix[3, 3] = 0.0
    
    bg = torch.zeros(3, device=device)
    campos = torch.zeros(3, device=device)
    
    # Test 1: Without feature rendering (backward compatible)
    print("Test 1: Without feature rendering...")
    settings_no_feat = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=math.tan(fovx/2), tanfovy=math.tan(fovy/2),
        bg=bg, scale_modifier=1.0,
        viewmatrix=viewmatrix, projmatrix=projmatrix,
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False,
        include_feature=False
    )
    rasterizer = GaussianRasterizer(raster_settings=settings_no_feat)
    color, feat_out, radii = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacities,
        shs=shs, scales=scales, rotations=rotations
    )
    print(f"  color shape: {color.shape}")      # (3, H, W)
    print(f"  feat_out shape: {feat_out.shape}") # (0,) empty
    print(f"  radii shape: {radii.shape}")       # (N,)
    assert color.shape == (3, H, W), f"Expected (3, {H}, {W}), got {color.shape}"
    assert feat_out.numel() == 0, f"Expected empty tensor, got {feat_out.shape}"
    print("  PASSED!")
    
    # Test 2: With feature rendering
    print("\nTest 2: With feature rendering...")
    settings_feat = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=math.tan(fovx/2), tanfovy=math.tan(fovy/2),
        bg=bg, scale_modifier=1.0,
        viewmatrix=viewmatrix, projmatrix=projmatrix,
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False,
        include_feature=True
    )
    rasterizer_feat = GaussianRasterizer(raster_settings=settings_feat)
    color2, feat_out2, radii2 = rasterizer_feat(
        means3D=means3D, means2D=means2D, opacities=opacities,
        shs=shs, scales=scales, rotations=rotations,
        language_feature_precomp=language_features
    )
    print(f"  color shape: {color2.shape}")       # (3, H, W)
    print(f"  feat_out shape: {feat_out2.shape}")  # (256, H, W)
    print(f"  radii shape: {radii2.shape}")        # (N,)
    assert color2.shape == (3, H, W), f"Expected (3, {H}, {W}), got {color2.shape}"
    assert feat_out2.shape == (256, H, W), f"Expected (256, {H}, {W}), got {feat_out2.shape}"
    print("  PASSED!")
    
    # Test 3: Backward pass
    print("\nTest 3: Backward pass with feature gradients...")
    loss_color = color2.sum()
    loss_feat = feat_out2.sum()
    loss = loss_color + loss_feat
    loss.backward()
    
    print(f"  language_features.grad shape: {language_features.grad.shape}")
    print(f"  language_features.grad max: {language_features.grad.abs().max().item():.6f}")
    assert language_features.grad is not None, "Expected gradients for language_features"
    assert language_features.grad.shape == (N, 256), f"Expected ({N}, 256), got {language_features.grad.shape}"
    # Check that at least some gradients are non-zero (some Gaussians should be visible)
    n_nonzero = (language_features.grad.abs() > 0).any(dim=1).sum().item()
    print(f"  Gaussians with non-zero gradients: {n_nonzero}/{N}")
    print("  PASSED!")
    
    # Test 4: RGB should be identical with or without features
    print("\nTest 4: RGB consistency check...")
    max_diff = (color - color2).abs().max().item()
    print(f"  Max RGB diff (with vs without features): {max_diff:.8f}")
    assert max_diff < 1e-5, f"RGB outputs should match, but diff is {max_diff}"
    print("  PASSED!")
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    test_feature_rendering()
