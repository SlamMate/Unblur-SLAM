#!/usr/bin/env python3
"""Runtime contract checks for the pluggable deblurring frontends."""

import tempfile
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deblur_backends import CausalTorchScriptBackend, EVSSMBackend


class IdentityEVSSM(torch.nn.Module):
    def forward(self, image):
        return image


class LastFrame(torch.nn.Module):
    def forward(self, frames):
        return frames[:, -1]


def main():
    odd_image = torch.rand(1, 3, 17, 19)
    evssm = EVSSMBackend(IdentityEVSSM(), "cpu")
    evssm_output = evssm(odd_image)
    assert evssm_output.shape == odd_image.shape
    assert torch.allclose(evssm_output, odd_image)

    with tempfile.TemporaryDirectory() as root:
        checkpoint = Path(root) / "last_frame.pt"
        traced = torch.jit.trace(LastFrame(), torch.rand(1, 3, 3, 8, 8))
        traced.save(str(checkpoint))
        causal = CausalTorchScriptBackend(checkpoint, history=3, device="cpu")

        first = torch.rand(1, 3, 8, 8)
        assert torch.allclose(causal(first), first)
        assert len(causal.frames) == 1

        resized = torch.rand(1, 3, 10, 12)
        assert torch.allclose(causal(resized), resized)
        assert len(causal.frames) == 1

    print("deblur_backend_contracts=PASS")


if __name__ == "__main__":
    main()
