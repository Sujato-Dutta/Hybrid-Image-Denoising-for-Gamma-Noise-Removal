"""Independent author-reference fidelity check; reference code is test-only."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.uformer import Uformer, DropPath, window_partition, window_reverse
from src.camera_ready_protocol import write_json


def author_reference(path: Path):
    # The reference's timm dependency is only three basic helpers. Supply these
    # for testing so importing a full vision-training library is unnecessary.
    # Numerical parity runs in eval mode, with stochastic depth inactive.
    names = ('timm', 'timm.models', 'timm.models.layers')
    saved = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    layer = sys.modules['timm.models.layers']
    layer.DropPath = DropPath
    layer.to_2tuple = lambda x: x if isinstance(x, tuple) else (x, x)
    layer.trunc_normal_ = torch.nn.init.trunc_normal_
    try:
        spec = importlib.util.spec_from_file_location('author_uformer_test_reference', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def verify(reference_path: Path) -> dict:
    torch.set_num_threads(4)
    torch.manual_seed(17)
    reference = author_reference(reference_path).Uformer(
        img_size=256, embed_dim=16, depths=[2] * 9, win_size=8,
        token_projection='linear', token_mlp='leff', modulator=True,
    ).eval()
    native = Uformer(in_channels=3).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        expected = reference(image)
        actual = native.forward_core(image)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    maximum_error = float((actual - expected).abs().max())
    rgb_parameters = sum(p.numel() for p in native.parameters())
    assert rgb_parameters == sum(p.numel() for p in reference.parameters())
    grayscale = Uformer().eval()
    x = torch.rand(1, 1, 180, 180, requires_grad=True)
    output = grayscale(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in grayscale.parameters())
    with torch.no_grad():
        odd = grayscale(torch.rand(1, 1, 181, 173))
    assert odd.shape == (1, 1, 181, 173)
    # A zero correction head must preserve the image exactly, including padding.
    with torch.no_grad():
        grayscale.output_proj.proj[0].weight.zero_()
        grayscale.output_proj.proj[0].bias.zero_()
        torch.testing.assert_close(grayscale(x.detach()), x.detach(), rtol=0, atol=0)
    windows_input = torch.rand(2, 16, 24, 7)
    torch.testing.assert_close(window_reverse(window_partition(windows_input, 8), 8, 16, 24), windows_input, rtol=0, atol=0)
    report = {'reference_revision': '65fc970a8ffc09605faca74ed016ee93c9ad8a36',
              'reference_rgb_parameters': rgb_parameters,
              'native_grayscale_parameters': sum(p.numel() for p in grayscale.parameters()),
              'max_absolute_rgb_output_error': maximum_error,
              'strict_state_dict_match': True, 'grayscale_backward_finite': True,
              'odd_shape_supported': True, 'global_residual_verified': True,
              'padded_180_input_size': grayscale.padded_side(180, 180), 'torch_version': torch.__version__}
    write_json(ROOT / 'docs/UFORMER_FIDELITY_CHECK.json', report)
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'tmp/references/Uformer/model.py')
    args = parser.parse_args()
    verify(args.reference)
