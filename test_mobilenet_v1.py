import unittest

import torch

from mobilenet_v1 import PartitionAndBlockingModel


class MultiMapProtectedBranchTests(unittest.TestCase):
    def test_parameter_count_is_independent_of_block_size(self):
        counts = []
        for mask_side in (10, 16, 22, 28):
            model = PartitionAndBlockingModel(mask_side=mask_side, protected_pool_size=4)
            counts.append(sum(parameter.numel() for parameter in model.protected_layers.parameters()))
        self.assertEqual(counts, [17_060] * 4)

    def test_parameter_count_for_configurable_pool_sizes(self):
        expected = {2: 2_658, 4: 17_060, 8: 131_880, 16: 1_049_648, 32: 8_390_208}
        for pooled_size, parameter_count in expected.items():
            model = PartitionAndBlockingModel(mask_side=10, protected_pool_size=pooled_size)
            self.assertEqual(sum(parameter.numel() for parameter in model.protected_layers.parameters()), parameter_count)

    def test_ordinary_view_hides_only_the_protected_block(self):
        model = PartitionAndBlockingModel(mask_side=10, mask_fill="zero")
        smashed = torch.ones(1, 32, 32, 32)
        visible = model.mask_central_partition(smashed)
        self.assertEqual(torch.count_nonzero(visible[:, :, 11:21, 11:21]).item(), 0)
        self.assertEqual(torch.count_nonzero(visible).item(), 32 * (32 * 32 - 10 * 10))

    def test_learned_fill_is_input_independent(self):
        model = PartitionAndBlockingModel(mask_side=10, mask_fill="learned")
        with torch.no_grad():
            model.mask_fill.copy_(torch.arange(32).view(1, 32, 1, 1))
        first = model.mask_central_partition(torch.randn(1, 32, 32, 32))
        second = model.mask_central_partition(torch.randn(1, 32, 32, 32))
        expected = model.mask_fill.expand(1, 32, 10, 10)
        self.assertTrue(torch.equal(first[:, :, 11:21, 11:21], expected))
        self.assertTrue(torch.equal(second[:, :, 11:21, 11:21], expected))

    def test_initial_filters_preserve_then_group_channels(self):
        model = PartitionAndBlockingModel(mask_side=10, protected_pool_size=4)
        branch = model.protected_layers
        protected = torch.rand(2, 32, 10, 10)
        with torch.no_grad():
            filtered = branch.relu_spatial(branch.spatial_filter(protected))
            projected = branch.channel_projection(filtered)
        self.assertTrue(torch.allclose(filtered, protected, atol=1e-7))
        self.assertEqual(projected.shape, (2, 4, 10, 10))
        for channel in range(4):
            self.assertTrue(torch.allclose(projected[:, channel], protected[:, channel * 8:(channel + 1) * 8].mean(dim=1), atol=1e-6))

    def test_branch_and_model_output_shapes(self):
        model = PartitionAndBlockingModel(mask_side=28, protected_pool_size=4).eval()
        with torch.no_grad():
            smashed = model.client_model(torch.randn(2, 3, 32, 32))
            protected = model.extract_central_partition(smashed)
            protected_output = model.protected_layers(protected)
            logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(protected.shape, (2, 32, 28, 28))
        self.assertEqual(protected_output.shape, (2, 256))
        self.assertEqual(logits.shape, (2, 10))

    def test_invalid_pool_size_is_rejected(self):
        with self.assertRaises(ValueError):
            PartitionAndBlockingModel(mask_side=10, protected_pool_size=0)

    def test_pool_size_larger_than_block_is_supported(self):
        model = PartitionAndBlockingModel(mask_side=22, protected_pool_size=32).eval()
        with torch.no_grad():
            protected = torch.randn(2, 32, 22, 22)
            pooled = model.protected_layers.pool(model.protected_layers.channel_projection(model.protected_layers.relu_spatial(model.protected_layers.spatial_filter(protected))))
            output = model.protected_layers(protected)
        self.assertEqual(pooled.shape, (2, 32, 32, 32))
        self.assertEqual(output.shape, (2, 256))


if __name__ == "__main__":
    unittest.main()
