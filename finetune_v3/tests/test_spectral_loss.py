"""CPU-only staged equivalence tests. No checkpoints, training or CUDA calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import unittest
import torch
from torch.nn import functional as F
from reference_loss import reference_spectral_loss
from finetune_v3 import training as candidate

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


class CPUEquivalenceTests(unittest.TestCase):
    def test_padding_exact_values_and_integer_weight_gradient(self):
        for padding in (0, 1, 2, 128, 256, 512):
            for extra in (1, 3, 521):
                for batch in (1, 2):
                    with self.subTest(padding=padding, extra=extra, batch=batch):
                        n = padding + extra
                        x = torch.arange(batch*n, dtype=torch.float64).reshape(batch, n).requires_grad_()
                        actual = candidate._reflect_center_pad(x, padding)
                        reference = F.pad(x, (padding, padding), mode='reflect')
                        self.assertTrue(torch.equal(actual, reference))
                        upstream = torch.arange(actual.numel(), dtype=torch.float64).reshape_as(actual)
                        grad = torch.autograd.grad(actual, x, upstream)[0]
                        refgrad = torch.autograd.grad(reference, x, upstream)[0]
                        expected = torch.zeros_like(x)
                        for j in range(n + 2*padding):
                            original = padding-j if j < padding else j-padding if j < padding+n else 2*n+padding-2-j
                            expected[:, original] += upstream[:, j]
                        self.assertTrue(torch.equal(grad, expected))
                        self.assertTrue(torch.equal(grad, refgrad))

    def test_padding_double_gradcheck_and_gradgradcheck(self):
        x = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
        fn = lambda wave: candidate._reflect_center_pad(wave, 4)
        self.assertTrue(torch.autograd.gradcheck(fn, (x,)))
        self.assertTrue(torch.autograd.gradgradcheck(fn, (x,)))

    def test_invalid_boundary_shapes(self):
        for size, padding in ((0, 0), (1, 1), (128, 128), (512, 512), (9, -1)):
            with self.subTest(size=size, padding=padding), self.assertRaises(ValueError):
                candidate._reflect_center_pad(torch.zeros(1, size), padding)
        with self.assertRaises(ValueError):
            candidate._reflect_center_pad(torch.zeros(1, 2, 8), 2)

    def test_stft_complex_exact_at_all_fixed_sizes(self):
        generator = torch.Generator().manual_seed(2026)
        for fft in (256, 512, 1024):
            for length in (fft//2+1, fft+13, 3600):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(fft=fft, length=length, dtype=dtype):
                        x = torch.randn(1, length, generator=generator, dtype=dtype)
                        window = torch.hann_window(fft, dtype=dtype)
                        reference = torch.stft(x, fft, hop_length=fft//4, window=window, return_complex=True)
                        actual = torch.stft(candidate._reflect_center_pad(x, fft//2), fft,
                                            hop_length=fft//4, window=window, center=False, return_complex=True)
                        self.assertTrue(torch.equal(actual, reference))

    def test_spectral_objective_exact_and_gradient_finite(self):
        reference = reference_spectral_loss
        generator = torch.Generator().manual_seed(2026)
        for length in (513, 1024, 3600):
            for kind in ('random', 'zero', 'impulse'):
                with self.subTest(length=length, kind=kind):
                    x = torch.randn(1, length, generator=generator) * .1
                    target = torch.randn(1, length, generator=generator) * .1
                    if kind == 'zero':
                        x.zero_()
                    if kind == 'impulse':
                        x.zero_()
                        x[0, 0] = .1
                    x.requires_grad_()
                    loss = candidate.spectral_loss(x, target)
                    ref = reference(x, target)
                    self.assertTrue(torch.equal(loss, ref))
                    grad = torch.autograd.grad(loss, x)[0]
                    self.assertTrue(torch.isfinite(grad).all())
                    names, pending = set(), [loss.grad_fn]
                    while pending:
                        node = pending.pop()
                        if node is None or node in names:
                            continue
                        names.add(node)
                        pending.extend(next_node for next_node, _ in node.next_functions)
                    self.assertFalse(any('ReflectionPad' in node.name() for node in names))

    def test_strict_cpu_profile_no_cuda_context(self):
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
