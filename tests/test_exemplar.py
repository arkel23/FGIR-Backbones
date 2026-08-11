"""The testing bar for this repo (see CLAUDE.md): flat tests, one behavior each,
arrange-act-assert inline, explicit tolerances, CPU-only, no downloads. Run from the
repo root (needs timm 0.6.x):

    python -m unittest tests.test_exemplar -v
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgir_backbones.model_utils.build_model import build_model


def tiny_args(**overrides):
    # The repo's own vit_t16 at 2 blocks: no timm model, no download, sub-second on CPU.
    args = SimpleNamespace(
        model_name="vit_t16", image_size=64, num_classes=10, pretrained=False,
        ckpt_path=None, transfer_learning=False, freeze_backbone=False,
        unfreeze_first_conv=False, distributed=False, device="cpu", selector=None,
        classifier=None, class_proj_size=256, sd=0.0, patch_stride=None,
        num_hidden_layers=2, seq_len=None)
    vars(args).update(overrides)
    return args


class TestExemplar(unittest.TestCase):
    def test_model_builds_and_forward_shape_no_nans(self):
        torch.manual_seed(0)
        model = build_model(tiny_args()).eval()
        y = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(y.shape, (2, 10))
        self.assertFalse(torch.isnan(y).any(), "forward produced NaNs")

    def test_gradients_flow_to_every_trainable_param(self):
        torch.manual_seed(1)
        model = build_model(tiny_args()).train()
        model(torch.randn(2, 3, 64, 64)).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} got no gradient")
                self.assertGreater(p.grad.pow(2).sum().item(), 0, f"{name} gradient is zero")

    def test_freeze_backbone_leaves_only_head_trainable(self):
        torch.manual_seed(2)
        model = build_model(tiny_args(freeze_backbone=True))
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable, "freezing left nothing trainable")
        for name in trainable:
            self.assertIn("head", name, f"{name} should be frozen")

    def test_batch_independence(self):
        torch.manual_seed(3)
        model = build_model(tiny_args()).eval()
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        model(x)[0].sum().backward()  # loss from sample 0 only
        self.assertGreater(x.grad[0].abs().sum().item(), 0)
        torch.testing.assert_close(x.grad[1], torch.zeros_like(x.grad[1]), rtol=0, atol=0)

    def test_single_optimizer_step_reduces_loss(self):
        torch.manual_seed(4)
        model = build_model(tiny_args()).eval()  # eval: dropout off, loss change is pure SGD
        x, target = torch.randn(2, 3, 64, 64), torch.tensor([1, 7])
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        before = torch.nn.functional.cross_entropy(model(x), target)
        before.backward()
        opt.step()
        after = torch.nn.functional.cross_entropy(model(x), target)
        self.assertLess(after.item(), before.item())

    def test_save_reload_roundtrip_outputs_identical(self):
        torch.manual_seed(5)
        model = build_model(tiny_args()).eval()
        fresh = build_model(tiny_args()).eval()
        missing = fresh.load_state_dict(model.state_dict())
        self.assertEqual(list(missing.missing_keys), [])
        x = torch.randn(2, 3, 64, 64)
        # Bit-identical: the reloaded model must be the same model, not an approximation.
        torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)

    def test_inference_deterministic_given_seed(self):
        outs = []
        for _ in range(2):
            torch.manual_seed(6)
            model = build_model(tiny_args()).eval()
            outs.append(model(torch.randn(2, 3, 64, 64)))
        torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
