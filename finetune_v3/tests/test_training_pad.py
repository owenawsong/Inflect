"""CPU-only training-pad contract tests; no inference runtime or weight edits."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import ast
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
ROOT = Path(os.environ.get("INFLECT_V3_REFERENCE_SOURCE_ROOT", str(Path(__file__).resolve().parents[2])))

from finetune_v3 import model as module
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def fixture(pad=None):
    holder = nn.Module()
    holder.generator = nn.Module()
    holder.generator.reflection_pad = pad if pad is not None else nn.ReflectionPad1d((1, 0))
    holder.generator.register_parameter('weight', nn.Parameter(torch.arange(3, dtype=torch.float32)))
    holder.other_pad = nn.ReflectionPad1d((1, 0))
    return holder


class TrainingPadTests(unittest.TestCase):
    def test_forward_and_analytic_gradient_exact(self):
        for n in (2, 3, 17, 7320):
            for shape in ((3, n), (1, 128, n)):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(shape=shape, dtype=dtype):
                        x = torch.arange(torch.tensor(shape).prod(), dtype=dtype).reshape(shape).requires_grad_()
                        actual = module._TrainingLeftReflectionPad1d()(x)
                        reference = nn.ReflectionPad1d((1, 0))(x)
                        self.assertTrue(torch.equal(actual, reference))
                        upstream = torch.arange(actual.numel(), dtype=dtype).reshape_as(actual)
                        a = torch.autograd.grad(actual, x, upstream)[0]
                        b = torch.autograd.grad(reference, x, upstream)[0]
                        expected = upstream[..., 1:].clone()
                        expected[..., 1] += upstream[..., 0]
                        self.assertTrue(torch.equal(a, expected))
                        self.assertTrue(torch.equal(a, b))

    def test_double_gradcheck_and_gradgradcheck(self):
        x = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
        layer = module._TrainingLeftReflectionPad1d()
        self.assertTrue(torch.autograd.gradcheck(layer, (x,)))
        self.assertTrue(torch.autograd.gradgradcheck(layer, (x,)))

    def test_invalid_shapes_fail(self):
        layer = module._TrainingLeftReflectionPad1d()
        for shape in ((1,), (3, 1), (1, 2, 0), (1, 2, 3, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                layer(torch.zeros(shape))

    def test_exact_path_only_empty_state_and_parameter_counts(self):
        holder = fixture().eval()
        previous = {key: value.clone() for key, value in holder.state_dict().items()}
        count = sum(p.numel() for p in holder.parameters())
        other = holder.other_pad
        module._install_training_generator_padding(holder)
        self.assertIs(type(holder.generator.reflection_pad), module._TrainingLeftReflectionPad1d)
        self.assertFalse(holder.generator.reflection_pad.training)
        self.assertIs(holder.other_pad, other)
        self.assertEqual(list(previous), list(holder.state_dict()))
        self.assertEqual(count, sum(p.numel() for p in holder.parameters()))
        self.assertTrue(all(torch.equal(value, holder.state_dict()[key]) for key, value in previous.items()))
        self.assertFalse(holder.generator.reflection_pad.state_dict())

    def test_wrong_path_padding_type_or_subclass_fail_without_mutation(self):
        class Subclass(nn.ReflectionPad1d):
            pass
        for pad in (nn.ReflectionPad1d((0, 1)), nn.ReflectionPad1d((1, 1)),
                    nn.ReplicationPad1d((1, 0)), Subclass((1, 0))):
            with self.subTest(pad=type(pad).__name__, padding=pad.padding):
                holder = fixture(pad)
                with self.assertRaises(ValueError):
                    module._install_training_generator_padding(holder)
                self.assertIs(holder.generator.reflection_pad, pad)
        holder = nn.Module()
        holder.unrecognized = fixture().generator
        with self.assertRaises(ValueError):
            module._install_training_generator_padding(holder)

    def test_unexpected_state_hooks_and_repeat_install_fail(self):
        for kind in ('parameter', 'buffer', 'null_parameter', 'child', 'forward_hook', 'pre_hook', 'backward_hook', 'backward_pre_hook', 'forward_override'):
            with self.subTest(kind=kind):
                holder = fixture()
                layer = holder.generator.reflection_pad
                if kind == 'parameter':
                    layer.register_parameter('extra', nn.Parameter(torch.ones(1)))
                elif kind == 'buffer':
                    layer.register_buffer('extra', torch.ones(1))
                elif kind == 'null_parameter':
                    layer.register_parameter('extra', None)
                elif kind == 'child':
                    layer.extra = nn.Identity()
                elif kind == 'forward_hook':
                    layer.register_forward_hook(lambda *args: None)
                elif kind == 'pre_hook':
                    layer.register_forward_pre_hook(lambda *args: None)
                elif kind == 'backward_hook':
                    layer.register_full_backward_hook(lambda *args: None)
                elif kind == 'backward_pre_hook':
                    layer.register_full_backward_pre_hook(lambda *args: None)
                else:
                    layer.forward = lambda x: x
                with self.assertRaises(ValueError):
                    module._install_training_generator_padding(holder)
                self.assertIs(holder.generator.reflection_pad, layer)
        holder = fixture()
        module._install_training_generator_padding(holder)
        with self.assertRaises(ValueError):
            module._install_training_generator_padding(holder)

    def test_public_runtime_factory_function_source_is_unchanged(self):
        original = (ROOT/'finetune_v3/model.py').read_text(encoding='utf-8')
        candidate = Path(module.__file__).read_text(encoding='utf-8')
        def function(text, name):
            tree = ast.parse(text)
            return next(ast.get_source_segment(text, node) for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        self.assertEqual(function(original, 'load_public_runtime'), function(candidate, 'load_public_runtime'))
        self.assertEqual(function(original, '_load_native'), function(candidate, '_load_native'))

    def test_no_cuda_context_and_no_global_pad_replacement(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertIs(nn.ReflectionPad1d, type(nn.ReflectionPad1d((1, 0))))


if __name__ == '__main__':
    unittest.main()
